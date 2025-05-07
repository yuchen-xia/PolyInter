import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from opencood.models.fuse_modules.fusion_in_one import regroup
# from opencood.models.heter_pyramid_nego import HeterPyramidNego
# from opencood.models.heter_pyramid_consis import HeterPyramidConsis
# from opencood.models.heter_pyramid_consis import coords_bev_pos_emdbed
from opencood.models.sub_modules.keyfeat_modules import KeyfeatAligner, KeyfeatAlignPerdim, Gate2PubSemantic
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.resize_net import ResizeNet
import torchvision
from opencood.models.fuse_modules.wg_fusion_modules import Converter
from opencood.tools.feature_show import feature_show


def coords_bev_pos_emdbed(H_size, W_size, num_pos_feats):
    
    # Relative dis of H and W to center
    meshgrid = [[0, H_size - 1, H_size], [0, W_size - 1, W_size]]
    batch_H, batch_W = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid], indexing='ij')
    scale = max(H_size, W_size) / 2
    batch_H = torch.abs((batch_H + 0.5) - H_size // 2) / scale
    batch_W = torch.abs((batch_W + 0.5) - W_size // 2) / scale
    coord_base = torch.cat([batch_H[None], batch_W[None]], dim=0) # (2, H, W)
    coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
    
    
    scale = 2 * math.pi
    pos = coord_base * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats + 1 # 避免除0
    pos_H = pos[..., 0, None] / dim_t
    pos_W = pos[..., 1, None] / dim_t
    pos_H = torch.stack((pos_H[..., 0::2].sin(), pos_H[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_W = torch.stack((pos_W[..., 0::2].sin(), pos_W[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_H, pos_W), dim=-1)  # (H*W, 2 * num_pos_feats)
    
    return posemb

class ComminPub(nn.Module):
    def __init__(self, args, pub_codebook=None, init_codebook=None):
        super(ComminPub, self).__init__()
        
        self.num_codes = args['num_codes']
        self.dim_code = args['dim']
        self.comm_space = args['comm_space']
        
        """
        dim: [num_codes, dim_code(c)]
        code-1: value-1
        code-2: value-2
        ...
        code-N: value-N
        """
        
        if 'modality_name' in args.keys():
            self.modality_name = args['modality_name']
        
        self.local_range = args['local_range']
        self.inference = args['inference'] if 'inference' in args.keys() else False
        args_unify =  args['unify_parameters']
        self.modality = args['modality']
        
        self.w_sigmod = True
        if 'w_sigmod' in args.keys():
            self.w_sigmod = args['w_sigmod']
        # unify_range = args_unify['unify_range']
        # self.unify_granularity = ((unify_range[4]-unify_range[1])/args_unify['H_uni'],
        #                           (unify_range[3]-unify_range[0])/args_unify['W_uni'])
        self.unify_granularity_H = args_unify['granularity_H']
        self.unify_granularity_W = args_unify['granularity_W']
        self.unify_channel = args_unify['C_uni']
        # self.cosim_threhold = args_unify['cosim_threhold']
        
        # 将初始特征映射为标准尺寸
        self.local2unify = ResizeNet(args['dim'], self.unify_channel, args['resizer']['reduce_raito'])
        
        # 发送端特征重组器, 将特征尺寸和通道数对齐到Common Space
        self.recombiner_send = AlignNet(args['recombiner'])
        
        # 特征选择器, 从重组后的特征中筛选出关键信息
        # self.foreground_selector = ForegroundSelector(args)
        
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.dim_code))
        
        self.H_size, self.W_size = (
            int((self.local_range[4] - self.local_range[1]) /  args_unify['granularity_H']), 
            int((self.local_range[3] - self.local_range[0]) /  args_unify['granularity_W'])
            )
        
        # self.bev_posemb = coords_bev_pos_emdbed(self.H_size, self.W_size, self.num_codes)
        # self.local_query_embeddings = nn.ModuleList()
        # for _ in range(self.num_codes):
        #     self.local_query_embeddings.append(nn.Sequential(
        #     nn.Linear(self.num_codes * 2, self.num_codes),
        #     nn.Linear(self.num_codes, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1, 1)
        # ))
        
        # args_filler = args['filler']
        # self.filler = nn.ModuleList()
        # for _ in range(self.num_codes):
        #     self.filler.append(
        #         Filler(args_filler)
        #     )
        # args_filler = args['filler']
        # args_filler = {"depth": 1, "input_dim": 1, "heads": 1, "dim_head": 1, "window_size":2}
        # self.filler = VallianceFiller(args_filler)
        
        if init_codebook is not None:
            self.codebook.data = init_codebook.data[:self.num_codes]
        # self.codebook = nn.Parameter(torch.arange(0, self.num_codes).unsqueeze(1).repeat(1, self.dim_code).to(torch.float))
        
        # self.weight_generator = nn.Parameter(torch.randn(self.dim_code))
        
        # self.translator_send = nn.Conv1d(3, 1, 1)
        # self.translator_receiver = nn.Conv1d(3, 1, 1)
        
        # self.translator_send = nn.Parameter(torch.randn(3, 1))
        
        # self.translator_receive = nn.Parameter(torch.randn(3, 1))
        
        # 初始化Tranlaotr, 使权重之间为恒等映射 
        # self.translator_send = nn.Parameter(torch.Tensor([[1], [0], [0]]))
        
        # self.translator_receive = nn.Parameter(torch.Tensor([[1], [0], [0]]))
        
        
        # self.translator_send = nn.Parameter(torch.ones(self.num_codes))
        
        # self.translator_receive = nn.Parameter(torch.ones(self.num_codes))
        
        self.enhancer_send = Converter(args['enhancer'])
            
        self.enhancer_receive = Converter(args['enhancer'])


        # self.negotiator_send = KeyfeatAlignPerdim(args['keyfeat_aligner_perdim'])
        self.negotiator_send = KeyfeatAligner(args['keyfeat_aligner'])
        
        # self.gate1 = Gate2PubSemantic(args['gate2pub'])
        # self.gate2 = Gate2PubSemantic(args['gate2pub'])
        # self.gate3 = Gate2PubSemantic(args['gate2pub'])
        
        
        # self.keyfeat_aligner_send = KeyfeatAlignPerdim(args['keyfeat_aligner_perdim'])
        # self.keyfeat_aligner_receive = KeyfeatAlignPerdim(args['keyfeat_aligner_perdim'])
        self.keyfeat_aligner_send = KeyfeatAligner(args['keyfeat_aligner'])
        self.keyfeat_aligner_receive = KeyfeatAligner(args['keyfeat_aligner'])


        
        # 将标准尺寸特征映射到本地尺寸
        self.unify2local = ResizeNet(self.unify_channel, args['dim'], args['resizer']['reduce_raito'])

        self.recombiner_receive = AlignNet(args['recombiner'])

        # self.pub2_codebook = None
        self.local2pub = None
        self.pub2local = None
        
        # self._init_weights()
    
    
    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        
        for name, para in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(para, 0)
            else:
                if len(para.shape) > 2:
                    init_func(para)
                else:
                    nn.init.normal_(para)
        
        nn.init.constant_(self.keyfeat_aligner_send.mlp_head, 1)
        nn.init.constant_(self.keyfeat_aligner_receive.mlp_head, 1)

    # def init_codebook_from_pub_cd(self, pub_codebook):
    #     len_local = self.codebook.shape[0]
    #     len_pub = pub_codebook.shape[0]
    #     if len_local <= len_pub:
    #         self.codebook.data = pub_codebook[:len_local, :]
    #     else:
    #         self.codebook.data[:len_pub, :] = pub_codebook

    def to(self, device):
        super(ComminPub, self).to(device)
        self.bev_posemb = self.bev_posemb.to(device)

    def pub_change(self, local2pub, pub2local, pub_codebook):
        self.local2pub = local2pub
        self.pub2local = pub2local
        self.pub_codebook = pub_codebook
        
        """
        从pub2local中, 解析出pub和local对应的idx, 并分别保存在list中
        以pub的idx为主序, local的list中, 加入与pub idx对应的idx
        """
        len_pub = len(self.pub_codebook)
        ref_pub_idxs = []
        ref_local_idxs = []
        for pub_idx in range(len_pub):
            if pub_idx in self.pub2local.keys():
                local_idx = self.pub2local[pub_idx]
                ref_pub_idxs.append(pub_idx)
                ref_local_idxs.append(local_idx)
                
        self.ref_pub_idxs = ref_pub_idxs
        self.ref_local_idxs = ref_local_idxs
        
        
        """
        更新pub_codepub codebook到local codebook的映射
        cosims: Torch.tensor [local_idx1 - float, local_idx2 - float,... ]
            local与pub对应向量之间的余弦相似度, 若不存在对应的pub_code, 赋值为0
            
        deltas: Torch.tensor
            local与pub之间模的差值
        
        """
        # len_local = len(self.codebook)
        # cosims = []
        # deltas = []
        # for local_idx in range(len_local):
        #     local_code = self.codebook[local_idx]
        #     if local_idx in self.local2pub.keys():
        #         pub_idx = self.local2pub[local_idx]
        #         pub_code = self.pub_codebook[pub_idx]
        #     else:
        #         pub_code = torch.zeros_like(local_code)
            
        #     cosim = torch.dot(local_code, pub_code)
        #     delta = torch.abs(torch.norm(local_code, p=2) - \
        #                         torch.norm(pub_code, p=2))
        #     cosims.append(cosim.item())
        #     deltas.append(delta.item())
        
        # # Translator将本地weights映射为公共的表示
        # self.cosims = torch.tensor(cosims).to(pub_codebook.device)        
        # self.deltas = torch.tensor(deltas).to(pub_codebook.device)
        
        
        # self.translator_send.data = torch.Tensor([[1], [0], [0]]).to(pub_codebook.device)
        
        # self.translator_receive.data = torch.Tensor([[1], [0], [0]]).to(pub_codebook.device)

    def sender(self, feature, pub_query = None):
        """
        每个特征点按照基向量解耦, 得到dict和weight
        feature: [b, c, h, w]
        
        return
        weights: [b, num_codes, h, w]-float weight list at each feature point
        codes: [b, num_codes, h, w]-int index list in codebook
        sg_map: 显著图, 衡量经过重组的特征的每个点是否存在关键信息
        """
        # # Crop/Padding to make feature represent unify range
        # feature = self.cavrange_to_unify(feature)          
        
        # Record feature size before unified
        self.local_size = feature.size()[1:]
        
        # 计算local特征和unify之间特征点表示粒度的比例关系
        local_granularity = ((self.local_range[4]-self.local_range[1]) / self.local_size[1],
                              (self.local_range[3]-self.local_range[0]) / self.local_size[2])
        
        # ratio_H, ratio_W
        unify_ratio = (local_granularity[0] / self.unify_granularity_H,
                       local_granularity[1] / self.unify_granularity_W)
        
        # (C, ratio_H * H, ratio_W * W)
        self.unify_size = (self.unify_channel, int(unify_ratio[0]*self.local_size[1]), int(unify_ratio[1]*self.local_size[2]))
        
        
        if hasattr(self, 'modality_name'):
            # local_weight = weights[0]
            feature_show(feature[0], f'analysis/direct_unify/f_{self.modality_name}')

        
        # Align feature size to unified size
        feature = self.local2unify(feature, self.unify_size)
        
        B, C, H, W = feature.size() 
        
        # Recombine feature size to make it easy to decompose
        feature = self.recombiner_send(feature)     
        
        
        
        if not self.inference:
            self.fc_af_recombine_send = feature
        
        
        feature = self.enhancer_send(feature)

        if hasattr(self, 'modality_name'):
            # local_weight = weights[0]
            feature_show(feature[0], f'analysis/direct_unify/f_refine_{self.modality_name}')       
        
        # Select useful feature information to transmit
        # mask, sg_map = self.foreground_selector(feature)
        # feature = feature * mask
        
        # feature_show(feature[0], '/home/scz/HEAL/analysis/fc_masked', type = 'mean')
        
        # [b, c, h, w] -> [b, num_codes, c, h, w]
        feature_expand = feature.unsqueeze(1).expand(B, self.num_codes, C, H, W)
        
        # [num_codes, c] -> 1, [num_codes, c, 1, 1] -> [b, num_codes, c, h, w]
        codebooks = self.codebook.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, self.num_codes, C, H, W)
        
        # # Mutiple every channel with parameter to get code weight 
        # # [b, num_codes, c, h, w] * [1, 1, c, 1, 1] -> [b, num_codes, c, h, w] 
        # # [num_codes, c, h, w]: [fc_cp1, fc_cp2, ...]
        # weights = (feature_expand - codebooks) * self.weight_generator.view(1, 1, C, 1, 1)
        # # [b, num_codes, c, h, w] -> [b, num_codes, h, w]
        # weights = weights.sum(dim=2)
        
        # Direct compute weight
        weights = (feature_expand * codebooks).sum(dim=2)
        weights = weights / (codebooks * codebooks).sum(dim=2)
        
        # Used to train vector
        if not self.inference:
            self.fc_before_rbdc = feature.clone()
            self.fc_after_rbdc = torch.einsum('b n h w, n c -> b c h w', weights, self.codebook)

        # if not self.inference:
        self.keyfeat_ori = weights

        # Local Convert
        weights = self.negotiator_send(weights, weights)
        
        # dim_w = weights[0].view(self.num_codes, -1)
        # cor_mat = torch.mm(dim_w, dim_w.t()) / (torch.norm(dim_w, p=2, dim=1)**2)
        # print(cor_mat)
        

            
        # self.keyfeat_bf_align2_ori = weights
        
        
        # gate1 = self.gate1(weights)
        # feature_show(gate1[0], f'analysis/cip_m1m3_gate/gate1_{self.modality}')
            
        # weights = weights * self.gate1(weights)    
        
        if not self.inference:
            # self.keyfeat_bf_align2 = weights
            weights = self.direct_local_w_to_pub(weights)
            
        return weights
    
    def sender_align(self, keyfeat_bf_align_query):

        if not self.inference: 
            keyfeat_bf_align_query = self.direct_pub_w_to_local(keyfeat_bf_align_query)
        
        """Avg Convert"""
        weights = self.keyfeat_aligner_send(keyfeat_bf_align_query, self.keyfeat_ori)

        
        # gate2待议
        # weights = weights * self.gate2(weights)

        self.keyfeat_af_align = weights
               
        # print(self.modality, self.local2pub)
        # print(self.modality, 'send', self.keyfeat_aligner_send.mlp_head)

        # sigmod归一化, 考虑去掉？
        if self.w_sigmod:
            weights = torch.sigmoid(weights)

        # dim_w = weights[0].view(self.num_codes, -1)
        # cor_mat = torch.matmul(dim_w, dim_w.t())
        # print(cor_mat)
        
        if hasattr(self, 'modality_name'):
            # local_weight = weights[0]
            feature_show(weights[0], f'analysis/direct_unify/w_{self.modality_name}')
            # feature_show(weights[0], f'analysis/feature_maps/w_{self.modality_name}')
            # feature_show(pub_query, f'analysis/feature_maps/pub_query')
        
        # feature_show(feature[0], 'analysis/feature_maps', type = 'mean')
        
        if self.comm_space == 'direct_pub':
            weights = self.direct_local_w_to_pub(weights)
            
        if hasattr(self, 'modality_name'):
            # pub_weight = weights[0]
            feature_show(weights[0], f'analysis/feature_maps/w_pub_{self.modality_name}')

        return weights
    
    def receiver(self, weights, record_len=None, affine_matrix=None,  record_len_modality = None):
        """
        weights_pub : torch.tensor
            [b, num_codes, h, w]
        
        affine_matrix : torch.tensor
            [B(total), L(max_cav), L, 2, 3]
            
        record_len : list
            shape: (B(total))
        
        return:
        weights_af_trans : torch.tensor
            [b, num_codes, h, w]
        
        feature:  : torch.tensor
            [b, c, h, w]
        After reconstructed
        """
        # re_w = weights
        # feature_show(weights[0], f'/home/scz/HEAL/analysis/direct_unify/weights_receive')  

        if (not self.inference) or (self.comm_space == 'direct_pub'): # 仅推理状态对特征重组; 训练时, 输入是本地完整模态, 无须重组
            # weights = self.direct_pub_w_to_local(weights, record_len, affine_matrix)
            weights = self.direct_pub_w_to_local(weights)

            # if self.inference:
        weights = self.align_bak_local(weights, record_len, affine_matrix, \
                                       record_len_modality=record_len_modality)
        # F.mse_loss(re_w.sum(dim=1), weights.sum(dim=1))
        # feature_show(weights[0], f'/home/scz/HEAL/analysis/direct_unify/weights_receive_processed')  
        if hasattr(self, 'modality_name'):
            feature_show(weights[0], f'/home/scz/HEAL/analysis/direct_unify/weights_receive_{self.modality_name}')  
            feature_show(weights[1], f'/home/scz/HEAL/analysis/direct_unify/weights_receive_{self.modality_name}_neb')   
        # 根据weights重构特征
        # softmax归一化, 是否启用有待验证, 因为这样计算出来的特征值就还是分布在[0~1]之间
        # weights = torch.softmax(weights, dim=1)
        
        # feature_show(weights[0], '/home/scz/HEAL/analysis/weights_receive', type = 'mean')
        
        # feature_show(fc_after_rbdc[0], '/home/scz/HEAL/analysis/fc_after_rbdc', type = 'mean')
        

        
        # if not self.inference:
        #     # weights = self.keyfeat_aligner_receive(weights, self.keyfeat_af_align)
        #     weights = self.keyfeat_aligner_receive(weights, weights)
        # else:
        #     weights = self.keyfeat_aligner_receive(weights, weights)

        # weights = self.keyfeat_aligner_receive2(weights, weights)

        # if not self.inference:
        #     self.keyfeat_bf_align_reverse = weights
        
        # print(self.modality, 'receive', self.keyfeat_aligner_receive.mlp_head)
        # print('\n')

        # Reconstruction and recombine to initial space
        feature = torch.einsum('b n h w, n c -> b c h w', weights, self.codebook)
        
        
        # Convert feature type to local style
        feature = self.enhancer_receive(feature)
        
        if not self.inference:
            self.fc_bf_recombine_receive = feature
        
        
        feature = self.recombiner_receive(feature)
        
        """
        Add extra supervision here for recombined feature?
        """
        
        # Align feature size to local size
        feature = self.unify2local(feature, self.local_size)
        
        if hasattr(self, 'modality_name'):
            feature_show(feature[0], '/home/scz/HEAL/analysis/direct_unify/feature_after_send')
            feature_show(feature[1], '/home/scz/HEAL/analysis/direct_unify/feature_after_send_neb')
        
        # # Crop/Padding to make feature represent unify range
        # feature = self.cavrange_to_local(feature)
        
        return feature        



    def direct_local_w_to_pub(self, weights):
        B, _, H, W = weights.size()
        weights_trans = torch.zeros((B, self.pub_codebook.shape[0], H, W)).to(weights.device)
        weights_trans[:, self.ref_pub_idxs] = weights[:, self.ref_local_idxs]     
        
        return weights_trans


    def direct_pub_w_to_local(self, weights_pub):

        """
        在这里对权重进行预处理:
        1. 只选择本地码本能解析的权重
        2. 缺失的权重用ego weights补齐
        """
        
        """
        找到本地能识别的权重, 即有对应local码本的维度
        """
        B, _, H, W = weights_pub.size()
        
        if hasattr(self, 'modality_name'):
            feature_show(weights_pub[0], f'analysis/feature_maps/w_pub_receive_{self.modality_name}')
            feature_show(weights_pub[1], f'analysis/feature_maps/w_pub_receive_m3')
        
        weights_received = torch.zeros((B, self.num_codes, H, W)).to(weights_pub.device)

        weights_received[:, self.ref_local_idxs] = weights_pub[:, self.ref_pub_idxs]
        
        if hasattr(self, 'modality_name'):
            # feature_show(local_query[0], f'analysis/feature_maps/local_query_{self.modality_name}')
            feature_show(self.query_local_wo_emb, f'analysis/feature_maps/local_query_wo_emb_{self.modality_name}')
            feature_show(weights_received[0], f'analysis/feature_maps/w_receive_{self.modality_name}')
            feature_show(weights_received[1], f'analysis/feature_maps/w_receive_m3')
        
        return weights_received
    
        
    def align_bak_local(self, weights_received, record_len, affine_matrix, record_len_modality = None):    
        """
        使用启用aligner将pub特征映射回本地
        若特征存在缺失的维度, 用ego的权重补齐
        """
        # _, _, H, W = weights_receive.size()
        _, C, H, W = weights_received.shape # C the number of local codebook
        
        B, _ = affine_matrix.shape[:2] # Different for pervious B, B_pre = sum(B*num_cav)
        # print(record_len)
        # b, c, h, w -> b, l, c, h, w
        
        # local_weights = regroup(self.keyfeat_af_align_ori, record_len)
        # ego的idx通过record_len获得, 还是record_len_modality获得
        if not self.inference:
            local_weights = regroup(self.keyfeat_af_align, record_len)
            
        elif self.inference and self.comm_space == 'direct_pub':
            local_weights = regroup(self.keyfeat_af_align, record_len_modality)
            assert self.keyfeat_af_align.shape[0] == sum(record_len_modality)
            
        elif self.inference and ('m' in self.comm_space):
            local_weights = regroup(self.keyfeat_af_align, record_len)
            assert self.keyfeat_af_align.shape[0] == sum(record_len)
            
        weights_received = regroup(weights_received, record_len)
        
        combined_weights = []
        for b in range(B):
            
            ego_weight = local_weights[b][0]
            # feature_show(ego_weight, f'analysis/ego_query_neb/ego_divide_by_modality')
            batch_receive_weight = weights_received[b]
            # feature_show(batch_receive_weight[0], f'analysis/ego_query_neb/ego_divide_by_record_len')

            
            N = record_len[b]
            batch_weights = []
            t_matrix = affine_matrix[b][:N, :N, :, :]
            i = 0

            warp_ego_weight = warp_affine_simple(ego_weight.unsqueeze(0).expand(N, -1, -1, -1),
                    t_matrix[:,i,:,:], (H, W))
            for cav_id in range(N):
                """local message"""
                # warp_ego_weight = warp_affine_simple(ego_weight.unsqueeze(0),
                #                     t_matrix[cav_id,i,:,:].unsqueeze(0), (H, W)).squeeze(0)
                # feature_show(warp_ego_weight[cav_id], f'analysis/ego_query_neb/ego_warpped_to{cav_id}')
                warp_ego_weight_cav = warp_ego_weight[cav_id]

                """receive messge"""
                cav_receive_weight = batch_receive_weight[cav_id] # C(self.num_codes) H W
                missing_dims = torch.where(torch.sum(cav_receive_weight, dim=(1, 2)) == 0)[0].tolist()
                
                if len(missing_dims) > 0:
                    # exist_dims = list(set(range(C)) - set(missing_dims))
                    # mean_exist_receive = torch.mean(cav_receive_weight[exist_dims], dim=0, keepdim=True)
                    
                    # feature_show(mean_exist_receive, \
                    #         f'analysis/atten_fill/neb_mean_w_of_cav_{cav_id}')
                    
                    for dim_idx in missing_dims: 
                        # cav_receive_weight[dim_idx] = \
                        #     self.filler(warp_ego_weight[dim_idx].unsqueeze(0).unsqueeze(0), \
                        #     mean_exist_receive.unsqueeze(0)).squeeze(0)
                        cav_receive_weight[dim_idx] = warp_ego_weight_cav[dim_idx]
                        
                        # feature_show(warp_ego_weight[dim_idx].unsqueeze(0), \
                        #     f'analysis/ego_query_neb/ego_w_used_to_fill_cav_{cav_id}_in_dim_{dim_idx}')
                            
                        # feature_show(cav_receive_weight[dim_idx].unsqueeze(0), \
                        #     f'analysis/ego_query_neb/neb_after_filled_of_cav_{cav_id}_in_dim_{dim_idx}')
                
                # feature_show(cav_receive_weight, f'analysis/ego_query_neb/ego_warpped_to{cav_id}')
                
                batch_weights.append(cav_receive_weight)
                
            batch_weights = torch.stack(batch_weights, dim=0)
            
            # batch_weights = batch_weights * self.gate3(batch_weights)
            
            # ego 查询 neb, 实现特征转换
            batch_weights = self.keyfeat_aligner_receive(warp_ego_weight, batch_weights)
            
            # ego 的 key feat 使用本地的
            batch_weights_lo = torch.cat((ego_weight.unsqueeze(0), batch_weights[1:]), dim=0)
            
            combined_weights.append(batch_weights_lo)
            
        combined_weights = torch.cat(combined_weights, dim=0)
        # combined_weights = torch.softmax(combined_weights, dim=1) # 最终选择去掉, 因此此时的weights已经是特征, 再使用softmax会修改特征信息
        
        # feature_show(combined_weights[0], '/home/scz/HEAL/analysis/feature_maps/combined_weights')
        # feature_show(combined_weights[1], '/home/scz/HEAL/analysis/feature_maps/combined_weights_neb')
        
        return combined_weights


    @property
    def query_local(self):
        query = []
        for c_idx in range(self.num_codes):
            dim_query = self.local_query_embeddings[c_idx](self.bev_posemb) # H*W, 1
            dim_query = dim_query.transpose(1, 0).view(1, self.H_size, self.W_size)
            query.append(dim_query)
        query = torch.cat(query, dim=0)
        return query
    
    @property
    def query_local_wo_emb(self):        
        query = self.bev_posemb.transpose(1, 0).view(self.num_codes * 2, self.H_size, self.W_size)
        return query