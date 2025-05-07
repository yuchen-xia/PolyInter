import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.resize_net import ResizeNet
import torchvision
from opencood.tools.feature_show import feature_show
    
class Comm(nn.Module):
    def __init__(self, args):
        super(Comm, self).__init__()
        
        self.num_codes = args['num_codes']
        self.dim_code = args['dim']
        
        """
        dim: [num_codes, dim_code(c)]
        code-1: value-1
        code-2: value-2
        ...
        code-N: value-N
        """
        
        
        
        self.local_range = args['local_range']
        args_unify =  args['unify_parameters']
        # unify_range = args_unify['unify_range']
        # self.unify_granularity = ((unify_range[4]-unify_range[1])/args_unify['H_uni'],
        #                           (unify_range[3]-unify_range[0])/args_unify['W_uni'])
        self.unify_granularity_H = args_unify['granularity_H']
        self.unify_granularity_W = args_unify['granularity_W']
        self.unify_channel = args_unify['C_uni']
        self.cosim_threhold = args_unify['cosim_threhold']
        
        # 将初始特征映射为标准尺寸
        self.local2unify = ResizeNet(args['dim'], self.unify_channel, args['resizer']['reduce_raito'])
        
        # 发送端特征重组器, 将特征尺寸和通道数对齐到Common Space
        self.recombiner_send = AlignNet(args['recombiner'])
        
        # 特征选择器, 从重组后的特征中筛选出关键信息
        # self.foreground_selector = ForegroundSelector(args)
        
        pub_cb_path = 'opencood/logs/pub_codebook/pub_codebook.pth'
        if 'pub_cb_path' in args_unify:
                pub_cb_path = args_unify['pub_cb_path']
        self.codebook_pub = torch.load(pub_cb_path)
        
        self.codebook = nn.Parameter(torch.randn(self.num_codes, self.dim_code))
        self.pub_cb_supervise = False
        if 'pub_cb_supervise' in args_unify:
            self.pub_cb_supervise = args_unify['pub_cb_supervise']
        self.pub_loaded = False
        # self.codebook = nn.Parameter(torch.arange(0, self.num_codes).unsqueeze(1).repeat(1, self.dim_code).to(torch.float))
        
        self.weight_generator = nn.Parameter(torch.randn(self.dim_code))
        
        # 本地到公共码本的字典
        # self.local_to_commom = torch.arange(self.num_codes-1, -1, -1)
        
        # 公共码本到本地的字典
        # self.common_to_local = torch.arange(self.num_codes-1, -1, -1)
        
        self.recombiner_receive = AlignNet(args['recombiner'])
        
        # 将标准尺寸特征映射到本地尺寸
        self.unify2local = ResizeNet(self.unify_channel, args['dim'], args['resizer']['reduce_raito'])

    def init_codebook_from_pub_cd(self, pub_codebook):
        len_local = self.codebook.shape[0]
        len_pub = pub_codebook.shape[0]
        if len_local <= len_pub:
            self.codebook.data = pub_codebook[:len_local, :]
        else:
            self.codebook.data[:len_pub, :] = pub_codebook
            

    def sender(self, feature):
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
        
        # 若使用pubcode监督训练, 并且未初始化codebook参数, 使用公共的codebook初始化本地codebook
        if self.training and self.pub_cb_supervise and (not self.pub_loaded): 
            self.init_codebook_from_pub_cd(self.codebook_pub)
            self.pub_loaded = True
        
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
        
        # Align feature size to unified size
        feature = self.local2unify(feature, self.unify_size)
        
        B, C, H, W = feature.size() 
        
        # Recombine feature size to make it easy to decompose
        feature = self.recombiner_send(feature)
        
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
        
        # sigmod归一化
        weights = torch.sigmoid(weights)
        
        
        # 将weights按照公共码本的顺序排列
        # weights = weights[:,self.local_to_commom, :, :]

        return weights, feature
    
    def receiver(self, weights, record_len, affine_matrix):
        """
        weights : torch.tensor
            [b, num_codes, h, w]
        
        affine_matrix : torch.tensor
            [B(total), L(max_cav), L, 2, 3]
            
        record_len : list
            shape: (B(total))
        
        return:
        feature:  : torch.tensor
            [b, c, h, w]
        After reconstructed
        """
        
        """
        在这里对权重进行预处理:
        1. 只保留本地能识别的特征
        2. 通过ego的完整weights为缺失基向量维度的nebs赋能? 实现特征重新组合
        
        实现
        首先提取出ego weights, 随后逐智能体依次判断, 为缺失的维度赋能
        """
        # 推理时, 只启用公共码本和本地码本共同包含的基向量, 并且将所有weights映射到ego视角下, 用于合并特征
        # if not self.training:
            # weights = self.seletc_fill_weight(weights, record_len, affine_matrix)
            
        # feature_show(weights[0], '/home/scz/HEAL/analysis/weights_partial', type = 'mean')
        
        # 将weights按照本地码本的顺序排列
        # weights = weights[:,self.common_to_local, :, :]
        

        
        # 根据weights重构特征
        # softmax归一化, 是否启用有待验证, 因为这样计算出来的特征值就还是分布在[0~1]之间
        weights = torch.softmax(weights, dim=1)
        fc_after_rbdc = torch.einsum('b n h w, n c -> b c h w', weights, self.codebook)
        
        # feature_show(fc_after_rbdc[0], '/home/scz/HEAL/analysis/fc_after_rbdc', type = 'mean')
        
        # Recombine to initial space
        feature = self.recombiner_receive(fc_after_rbdc)
        
        """
        Add extra supervision here for recombined feature?
        """
        
        # Align feature size to local size
        feature = self.unify2local(feature, self.local_size)
        
        # feature_show(feature[0], '/home/scz/HEAL/analysis/feature_after_send', type = 'mean')
        
        # # Crop/Padding to make feature represent unify range
        # feature = self.cavrange_to_local(feature)
        
        return feature, fc_after_rbdc
    

    def seletc_fill_weight(self, weights, record_len, affine_matrix):
        """
        weights : torch.tensor
            [b, num_codes, h, w]
        
        affine_matrix : torch.tensor
            [B(total), L(max_cav), L, 2, 3]
            
        record_len : list
            shape: (B(total))
        """

        self.codebook_pub = self.codebook_pub.to(self.codebook.device)
        # 模拟计算, 将本地不包含的code置为0计算local和public的codebook余弦相似度矩阵
        norm_local = F.normalize(self.codebook, p=2, dim=1)
        norm_pub = F.normalize(self.codebook_pub, p=2, dim=1)
        cosim_mat = torch.mm(norm_local, norm_pub.t()) # (len_local, len_pub)
        
        max_cosims, _ = torch.max(cosim_mat, dim=1)
        
        masked_wei_indcices = torch.where(max_cosims<=self.cosim_threhold)[0]
        # weights[:,masked_wei_indcices,:,:] = 0
        
        # 使用ego_weights填充neb中缺失的维度
        _, C, H, W = weights.shape
        B, L = affine_matrix.shape[:2]
        split_w = regroup(weights, record_len)
        # w_before = split_w[0][1].clone()
        
        # neb的非公共基向量置0
        for b in range(B):
            split_w[b][1:, masked_wei_indcices] = 0
            
        # w_middle = split_w[0][1].clone()
        # print(F.mse_loss(w_before, w_middle))
        
        
        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = affine_matrix[b][:N, :N, :, :]
            i = 0
            
            # ego的weights warp搭配neb视角下, 用于填充缺失基向量的权重
            ego_weights = split_w[b][0].repeat(N, 1, 1, 1)
            ego_weights = warp_affine_simple(ego_weights,
                                            t_matrix[:,i,:,:],
                                            (H, W))
            # feature_show(ego_weights[1], '/home/scz/HEAL/analysis/weight_fills', type = 'mean')
            
            # neb特征中, 缺失的基向用ego对应的weights填充
            w = split_w[b]
            # feature_show(w[1], '/home/scz/HEAL/analysis/weights1_before', type = 'mean')
            w_sum = w.sum(dim=(-1, -2))
            for n in range(1, N):
                # continue
                fill_idxs = torch.where(w_sum[n]==0, True, False)
                w[n][fill_idxs] = ego_weights[n][fill_idxs]
            out.append(w)
        # w_after = out[0][1].clone()    
        # print(F.mse_loss(w_after, w_before))
            # feature_show(w[0], '/home/scz/HEAL/analysis/weights_ego', type = 'mean')
            # feature_show(w[1], '/home/scz/HEAL/analysis/weights1_after', type = 'mean')
            
            
            # 将weights映射到ego视角下
            # ego_weights = split_w[b][0]
            # weights_in_ego = warp_affine_simple(split_w[b],
            #                                 t_matrix[i,:,:,:],
            #                                 (H, W))
            
            # # feature_show(ego_weights, '/home/scz/HEAL/analysis/init_weight', type = 'mean')
            # # feature_show(weights_in_ego[0], '/home/scz/HEAL/analysis/weights0', type = 'mean')
            # # feature_show(weights_in_ego[1], '/home/scz/HEAL/analysis/weights1', type = 'mean')
            
            # # neb特征中, 缺失的基向用ego对应的weights填充
            # weight_sum = weights_in_ego.sum(dim=(-1, -2))
            # for n in range(1, N):
            #     replace_idxs = torch.where(weight_sum[n]==0, True, False)
            #     weights_in_ego[n][replace_idxs] = ego_weights[replace_idxs]
                # weights_in_ego[n] = 0
                # replace_idxs_after = torch.where(weights_in_ego[n].sum(dim=(-1, -2))==0, True, False)
            # feature_show(weights_in_ego[1], '/home/scz/HEAL/analysis/weights1_added', type = 'mean')
            # out.append(weights_in_ego)
        
        weights_out = torch.cat(out, dim=0)
        
        return weights_out

    # def cavrange_to_unify(self, x):
    #     """
    #     Crop/Padding to make feature represent unify range
    #     """
    #     _, _, self.init_H, self.init_W = x.size() 
        
    #     self.crop_target_H = int(self.init_H * (self.unify_range[4] - self.unify_range[1]) \
    #                                   / (self.local_range[4] - self.local_range[1])) 
    #     self.crop_target_W = int(self.init_W * (self.unify_range[3] - self.unify_range[0]) \
    #                                   / (self.local_range[3] - self.local_range[0])) 
        
    #     crop_func = torchvision.transforms.CenterCrop((self.crop_target_H, self.crop_target_W))
    #     x_cropped = crop_func(x)
        
    #     # Get feature after cropped
    #     # 计算中心裁剪的起始和结束位置
    #     top = (self.init_H - self.crop_target_H) // 2
    #     left = (self.init_W - self.crop_target_W) // 2
    #     bottom = top + self.crop_target_H
    #     right = left + self.crop_target_W
    #     self.init_x_top = x[:, :, :top, :] if top > 0 else torch.tensor([])
    #     self.init_x_bottom = x[:, :, bottom:, :] if bottom < self.init_H else torch.tensor([])
    #     self.init_x_left = x[:, :, top:bottom, :left] if left > 0 else torch.tensor([])
    #     self.init_x_right = x[:, :, top:bottom, right:] if right < self.init_W else torch.tensor([])
        
        
    #     return x_cropped
    
    # def cavrange_to_local(self, x):
    #     """
    #     Crop/Padding feature from unify range to local range
    #     """
    #     # 组合被裁剪的特征
    #     if self.init_x_left.numel():
    #         x = torch.cat([self.init_x_left, x], dim=3)
    #     if self.init_x_right.numel():
    #         x = torch.cat([x, self.init_x_right], dim=3)
        
    #     if self.init_x_top.numel():
    #         x = torch.cat([self.init_x_top, x], dim=2)
    #     if self.init_x_bottom.numel():
    #         x = torch.cat([x, self.init_x_bottom], dim=2)
            
        
    #     # Pad/Crop to local range
    #     crop_func = torchvision.transforms.CenterCrop((self.init_H, self.init_W))
    #     x_cropped = crop_func(x)
        
    #     return x_cropped
    