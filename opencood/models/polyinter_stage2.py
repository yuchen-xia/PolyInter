import torch
import torch.nn as nn

from opencood.tools import train_utils
from opencood.models.sub_modules.compressor import simple_align
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.sub_modules.adapter import TransformerDecoder
import matplotlib.pyplot as plt
from opencood.models.polyinter_stage1 import CrossAttention, detect_head
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import CoBEVT


class PolyInterStage2(nn.Module):
    def __init__(self, args) -> None:
        super(PolyInterStage2, self).__init__()
        self.fusion_method="fcooper"
        if 'fusion_net' in args and 'core_method' in args['fusion_net']:
            self.fusion_method=args['fusion_net']['core_method']
        print(f"use {self.fusion_method} fusion method")
        
        self.sensor_type_q = args["encoder_q"]["sensor_type"]
        self.sensor_type_k = args["encoder_k"]["sensor_type"]
        
        # 一、二阶段均不训练，均要加载参数
        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        self.detect_head = detect_head(args=args['encoder_q']['args'])
        
        
        # 加载预训练参数（需要注意一下模型名的前缀是否匹配，灵活使用load_pretrained_model和load_pretrained_submodule）
        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        if self.sensor_type_k == "lidar":
            self.encoder_k = train_utils.load_pretrained_model(    
                args["encoder_k"]["saved_pth"], self.encoder_k
            )
        else:
            self.encoder_k = train_utils.load_pretrained_submodule(    
                args["encoder_k"]["saved_pth"], self.encoder_k, "encoder_m4" ## efficent-net是m2，resnet是m4, 其他都是encoder_k
            )
        
        
        self.detect_head = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.detect_head
        )
        
        for param in self.encoder_q.parameters():
            param.requires_grad = False
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.detect_head.parameters():
            param.requires_grad = False
        
        c_prompt = args["encoder_q"]["args"]["channel"]
        h_q, w_q = args["encoder_q"]["args"]['featrue_size']
        h_k, w_k = args["compressor_k"]['featrue_size']
        self.general_prompt_k = nn.Parameter(torch.load(args["pre_train_modules"])['general_prompt_k'],requires_grad=False).to('cuda')
        
        # 只一阶段训练，二阶段加载参数
        self.transformer = TransformerDecoder(args["transformer"])
        self.transformer = train_utils.load_pretrained_submodule(
            args["pre_train_modules"], self.transformer, "transformer"
        )
        self.channel_cross_attention = CrossAttention(h_q*w_q, c_prompt)
        self.channel_cross_attention = train_utils.load_pretrained_submodule(
            args["pre_train_modules"], self.channel_cross_attention, "channel_cross_attention"
        )
        if self.fusion_method=='fcooper':
            self.fusion_net = SpatialFusion()
        elif self.fusion_method=='cobevt':
            self.fusion_net = CoBEVT(args['fusion_net'])
        
        for param in self.channel_cross_attention.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.fusion_net.parameters():
            param.requires_grad = False
        
        # 一、二阶段均训练，不用加载参数
        self.compressor_k = simple_align(args['compressor_k'])
        self.specific_prompt_k_temp = torch.zeros(args["encoder_k"]["args"]["channel"], h_q, w_q)
        self.specific_prompt_k = nn.Parameter(self.specific_prompt_k_temp, requires_grad=True)
        torch.nn.init.xavier_normal_(self.specific_prompt_k, gain=1.0)    
        
        """For feature transformation"""
        self.cav_range = args['lidar_range']
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        
    def visualization(self, img, save_path='img.png', pooling='max'):
        if pooling == 'max':
            img = img.detach().squeeze(0).max(dim=0)[0].cpu().numpy()
        elif pooling == 'mean':
            img = img.detach().squeeze(0).mean(dim=0).cpu().numpy()
        elif pooling == 'channel':
            img = img.detach().squeeze(0)[200].cpu().numpy()
        plt.matshow(img, cmap=plt.cm.Reds)
        plt.savefig(save_path)
        return True


    def proj_neb2ego(self, neb_feat: torch.Tensor, 
                     ego2neb_t_matrix: torch.Tensor) -> torch.Tensor:
        """
        将neb feature投射到自车坐标下。
        
        Args:
            neb_feat:           [num_cav, C, H, W]
            ego2neb_t_matrix:   [1, max_cav, 2, 3] normalize后的t_matrix
        Returns:
            projected_neb_feat:  投影后的neb feature.
        """
        N, C, H, W = neb_feat.shape
        ego2neb_t_matrix = ego2neb_t_matrix[0, :N, :, :]
        neb_feat = warp_affine_simple(neb_feat, ego2neb_t_matrix, (H, W))
        
        return neb_feat        
   
    
    def forward(self, data_dict):
        """
        data_dict: 一个字典，输入数据
            包含如下key：
            1. "record_len": 原始数据中每个数据里有多少辆车。shape: [batch_size]
            2. "inputs_ego": 自车输入数据
            3. "inputs_neb": 邻车输入数据
            ...
        为了方便理解，我们必须说明：
            我们在数据处理的时候做了一些处理（数据处理过程见intermediate_heter_pair_polyinter_stage2_fusion_dataset.py）
            邻车为空的时候我们用自车数据补充了进去，防止它为空。
            然后我们将自车数据复制了n份，让自车和邻车的数量相等，这么做是为了后续好批量计算attention，而不用写循环。
        """
        
        record_len = data_dict["record_len"]
        neb_num_sum = 0
        for cav_num in record_len.tolist():
            neb_num = cav_num - 1
            if neb_num == 0:
                neb_num = 1
            neb_num_sum += neb_num
        data_dict["inputs_neb"]['neb_sum'] = neb_num_sum # 邻车数量和，vn和sd需要
        ego2neb_t_matrix = data_dict['ego2neb_t_matrix']
        ego2neb_t_matrix = normalize_pairwise_tfm(ego2neb_t_matrix, self.H, self.W, self.fake_voxel_size)
        with torch.no_grad():
            batch_dict_q = self.encoder_q(data_dict["inputs_ego"])
            batch_dict_k = self.encoder_k(data_dict["inputs_neb"])
            query=batch_dict_q["spatial_features_2d"] #[2, 256, 50, 176]
            if self.sensor_type_k == "lidar":
                key = batch_dict_k["spatial_features_2d"]
            else:
                key = batch_dict_k
        
        # dimension alignment
        key = self.compressor_k(key)
        specific_prompt_k = self.specific_prompt_k.expand(key.size(0),-1,-1,-1)
        specific_prompt_k = self.compressor_k.conv_downsampling(specific_prompt_k)

        ### project neb to ego coordinate
        i = 0
        projected_key = []
        for b, cav_num in enumerate(record_len):
            neb_num = cav_num - 1
            neb_num = 1 if neb_num == 0 else neb_num
            temp =  self.proj_neb2ego(key[i:i+neb_num], ego2neb_t_matrix[b])
            i += neb_num
            projected_key.append(temp)
        key = torch.cat(projected_key)

        # channel and spatial
        general_prompt_k = self.general_prompt_k.expand(key.size(0),-1,-1,-1)
        prompts = torch.cat([specific_prompt_k, general_prompt_k],dim=1)
        key = self.channel_cross_attention(prompts, query, key, 'k')
        specific_prompt = key[:,:specific_prompt_k.size(1),:,:]
        general_prompt = key[:,specific_prompt_k.size(1):,:,:]
        specific_prompt_out = self.transformer(specific_prompt, query)  

        
        
        # for style loss
        mean_k = torch.mean(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False)
        mean_k2 = torch.mean(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False)
        mean_k3 = torch.mean(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False)
        mean_q = torch.mean(query.contiguous().view(key.size(0), -1), dim=1, keepdim=False)
        
        std_k = torch.var(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_k2 = torch.var(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_k3 = torch.var(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_q = torch.var(query.contiguous().view(query.size(0), -1), dim=1, keepdim=False, unbiased=False)
        
        
        # for fusion
        spatial_features_2d=[]
        i = 0
        for data_idx in range(len(record_len)):
            spatial_features_2d.append(query[i:i+1])
            neb_num = record_len[data_idx] - 1
            if (neb_num != 0):
                spatial_features_2d.append(specific_prompt_out[i:neb_num+i])
                i = neb_num+i
            else:
                i+=1
        spatial_features_2d = torch.cat(spatial_features_2d,dim=0)
        assert spatial_features_2d.shape[0] == sum(record_len)
        
        
        # fusion
        if self.fusion_method=='fcooper':
            out = self.fusion_net(spatial_features_2d, record_len)

        if self.fusion_method=='cobevt':
            identical_matrix = torch.tile(torch.eye(4), (ego2neb_t_matrix.shape[0], 5, 5, 1, 1))
            identical_matrix = normalize_pairwise_tfm(identical_matrix, self.H, self.W, self.fake_voxel_size)
            out = self.fusion_net(spatial_features_2d, record_len, identical_matrix)

        # detection
        output_dict = self.detect_head(out)
        key_dict = self.detect_head(specific_prompt)
        output_dict.update({"mean_q":mean_q, "mean_k":mean_k, "std_q":std_q, "std_k":std_k})
        output_dict.update({"mean_k2": mean_k2, "mean_k3": mean_k3, "std_k2": std_k2, "std_k3":std_k3})

        output_dict['psm_key'] = key_dict['psm']
        output_dict['rm_key'] = key_dict['rm']
        output_dict['record_len'] = record_len
        return output_dict