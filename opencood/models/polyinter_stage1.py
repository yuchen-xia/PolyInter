import torch
import torch.nn as nn
import math

from opencood.tools import train_utils
from opencood.models.sub_modules.compressor import simple_align
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.sub_modules.adapter import TransformerDecoder
from opencood.models.da_modules.gradient_layer import GradientScalarLayer
from opencood.models.sub_modules.multihead_attention import MultiheadAttention
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fusion_in_one import CoBEVT


torch.autograd.set_detect_anomaly(True)



class CrossAttention(nn.Module):
    def __init__(self, in_features, in_channels, embed_dim=256, num_heads=1, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn1 = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp1 = nn.Linear(in_features=in_features, out_features=embed_dim)
        self.mlp2 = nn.Linear(in_features=in_features, out_features=embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(in_features)
        self.layer_norm4 = nn.LayerNorm(in_features)
        self.layer_norm5 = nn.LayerNorm(in_features)
        self.conv_k1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_k2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
    
    def forward(self, prompts, query, key, selected_char='k', mask=None):
        # query, key, value shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = query.shape
        prompts_channel = prompts.size(1)
        
        prompts1_conv = self.conv_k1(prompts[:,:query.size(1),:,:])
        prompts2_conv = self.conv_k1(prompts[:,query.size(1):,:,:])
        
        prompts = torch.cat([prompts1_conv, prompts2_conv], 1)
        
        key_conv = self.conv_k2(key)

        key_conv = key_conv.view(batch_size, channels, height * width)
        
        # Reshape to (height*width, batch_size, channels) for multihead attention
        prompts = prompts.view(batch_size, prompts_channel, height * width)
        query = query.view(batch_size, channels, height * width)
        key = key.reshape(batch_size, channels, height * width)
        query = self.layer_norm1(self.mlp1(query))
        key_k = self.layer_norm2(self.mlp2(key))
        # prompts = self.layer_norm6(self.mlp3(prompts))

        # Perform cross-attention
        prompts_output1, _ = self.multihead_attn1(query, key_k, prompts[:,:query.size(1),:])
        prompts_output2, _ = self.multihead_attn1(query, key_k, prompts[:,query.size(1):,:])
        prompts_output = torch.cat([prompts_output1, prompts_output2],1)
        prompts_output = self.layer_norm3(prompts + prompts_output)
        
        key_output = self.layer_norm4(key_conv + self.multihead_attn2(query, key_k, key_conv)[0])

        prompts_output[:,:query.size(1),:] += key_output
        prompts_output[:,query.size(1):,:] += key_output
        prompts_output = self.layer_norm5(prompts_output)
        
        prompts_output = prompts_output.view(batch_size, prompts_channel, height, width)
        # Reshape back to (batch_size, channels, height, width)
        
        return prompts_output


class detect_head(nn.Module):
    def __init__(self, args) -> None:
        super(detect_head, self).__init__()
        in_channel=args['channel']
        self.cls_head = nn.Conv2d(in_channel, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(
            in_channel, 7 * args["anchor_number"], kernel_size=1
        )

    def forward(self, x):
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        return {"psm": psm, "rm": rm}

class DomainClassifier(nn.Module):
    def __init__(self, args) -> None:
        super(DomainClassifier, self).__init__()
        self.conv_layer1 = nn.Conv2d(args['in_channel'], 64, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(64, 32, kernel_size=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        temp = (math.floor((args['in_size'][0]-10)/2)+1)*(math.floor((args['in_size'][1]-10)/2)+1)
        self.linear_layer = nn.Linear(temp, args['out_size'])

        self.rgl = GradientScalarLayer(-9.1)
    
    def forward(self, feature):
        feature = self.rgl(feature)
        feature = self.conv_layer1(feature)
        feature = torch.relu(feature)
        feature = self.conv_layer2(feature)
        feature = torch.relu(feature)
        feature = self.maxpool(feature)
        feature = feature.max(dim=1)[0]
        feature = feature.reshape(feature.size(0), -1)
        out = self.linear_layer(feature)
        return out




class PolyInterStage1(nn.Module):
    def __init__(self, args) -> None:
        super(PolyInterStage1, self).__init__()
        self.fusion_method="fcooper"
        if 'fusion_net' in args and 'core_method' in args['fusion_net']:
            self.fusion_method=args['fusion_net']['core_method']
        assert self.fusion_method=='fcooper' or self.fusion_method=='where2comm' or self.fusion_method=='cobevt' or self.fusion_method=='v2xvit'
        print(f"use {self.fusion_method} fusion method")
        
        self.sensor_type_q = args["encoder_q"]["sensor_type"]
        self.sensor_type_k = args["encoder_k"]["sensor_type"]
        self.sensor_type_v = args["encoder_v"]["sensor_type"]
        
        # 一、二阶段均不训练，均要加载参数
        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        self.encoder_v = train_utils.create_encoder(args["encoder_v"])
        self.detect_head = detect_head(args=args['encoder_q']['args'])

        
        
        # 加载预训练参数（需要注意一下模型名的前缀是否匹配，灵活使用load_pretrained_model和load_pretrained_submodule）
        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        self.encoder_k = train_utils.load_pretrained_submodule(    
            args["encoder_k"]["saved_pth"], self.encoder_k, "encoder_m2" ## efficent-net是m2，resnet是m4, 其他都是encoder_k
        )
        # self.encoder_k = train_utils.load_pretrained_model(    
        #     args["encoder_k"]["saved_pth"], self.encoder_k ## efficent-net是m2，resnet是m4, 其他都是encoder_k
        # )
        self.encoder_v = train_utils.load_pretrained_model(    
            args["encoder_v"]["saved_pth"], self.encoder_v
        )
        
        
        self.detect_head = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.detect_head
        )
        
        for param in self.encoder_q.parameters():
            param.requires_grad = False
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.encoder_v.parameters():
            param.requires_grad = False
        for param in self.detect_head.parameters():
            param.requires_grad = False
        
        c_prompt = args["encoder_q"]["args"]["channel"]
        h_q, w_q = args["encoder_q"]["args"]['featrue_size']
        h_k, w_k = args["compressor_k"]['featrue_size']
        h_v, w_v = args["compressor_v"]['featrue_size']
        self.general_prompt_k = nn.Parameter(torch.FloatTensor(1, c_prompt, h_k, w_k).to('cuda'), requires_grad=True)
        torch.nn.init.xavier_normal_(self.general_prompt_k, gain=1.0)
        
        # 只一阶段训练，二阶段加载参数
        self.transformer = TransformerDecoder(args["transformer"])
        self.domain_classifier = DomainClassifier(args['domain_classifier'])
        self.channel_cross_attention = CrossAttention(h_q*w_q, c_prompt)
        if self.fusion_method=='fcooper':
            self.fusion_net = SpatialFusion()
        if self.fusion_method=='cobevt':
            self.fusion_net = CoBEVT(args['fusion_net'])
        
        # 一、二阶段均训练，不用加载参数
        self.compressor_k = simple_align(args['compressor_k'])
        self.compressor_v = simple_align(args['compressor_v'])
        self.specific_prompt_k_temp = torch.zeros(args["encoder_k"]["args"]["channel"], h_q, w_q)
        self.specific_prompt_k = nn.Parameter(self.specific_prompt_k_temp, requires_grad=True)
        
        
        self.specific_prompt_v_temp = torch.zeros(args["encoder_v"]["args"]["channel"], h_q, w_q)
        self.specific_prompt_v = nn.Parameter(self.specific_prompt_v_temp, requires_grad=True)
        torch.nn.init.xavier_normal_(self.specific_prompt_k, gain=1.0)
        torch.nn.init.xavier_normal_(self.specific_prompt_v, gain=1.0)
        
        self.count = 0
        
        
        """For feature transformation"""
        self.cav_range = args['lidar_range']
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        
    
    # def forward_q(self, data_dict): ##只跑自车
    #     record_len = data_dict["record_len"]
    #     pairwise_t_matrix = data_dict['pairwise_t_matrix']
    #     with torch.no_grad():
    #         batch_dict_q = self.encoder_q(data_dict["inputs_ego"])
    #         query=batch_dict_q["spatial_features_2d"]
        
        
    #     i = 0
    #     q_out = []
    #     for data_idx in range(len(record_len)):
    #         q_out.append(query[i])
    #         neb_num = record_len[data_idx] - 1
    #         if (neb_num != 0):
    #             i = neb_num+i
    #         else:
    #             i+=1
    #     out = torch.stack(q_out)
    #     output_dict = self.detect_head(out)
    #     return output_dict
        
        
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
            我们在数据处理的时候做了一些处理（数据处理过程见intermediate_heter_pair_polyinter_stage1_fusion_dataset.py）
            邻车为空的时候我们用自车数据补充了进去，防止它为空。
            然后我们将自车数据复制了n份，让自车和邻车的数量相等，这么做是为了后续好批量计算attention，而不用写循环。
        """
        
        record_len = data_dict["record_len"]
        
        
        neb_num_sum = 0
        for cav_num in record_len.tolist():
            neb_num = cav_num - 1
            if neb_num == 0: # 这里是因为neb_num为0的时候我们补充了一个自车数据进去
                neb_num = 1
            neb_num_sum += neb_num
        data_dict["inputs_neb"]['neb_sum'] = neb_num_sum # this is for sd and vn
        
        
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        ego2neb_t_matrix = data_dict['ego2neb_t_matrix']
        ego2neb_t_matrix = normalize_pairwise_tfm(ego2neb_t_matrix, self.H, self.W, self.fake_voxel_size)
        
        # get sensor type
        if data_dict['selected_char'] == 'k':  # selected_char表示邻车是哪种车型
            sensor_type = self.sensor_type_k
        elif data_dict['selected_char'] == 'v':
            sensor_type = self.sensor_type_v
        with torch.no_grad():
            batch_dict_q = self.encoder_q(data_dict["inputs_ego"])
            if data_dict['selected_char'] == 'k':
                batch_dict_k = self.encoder_k(data_dict["inputs_neb"])
                
            elif data_dict['selected_char'] == 'v':
                batch_dict_k = self.encoder_v(data_dict["inputs_neb"])
            query=batch_dict_q["spatial_features_2d"] 

            if sensor_type == "lidar":
                key = batch_dict_k["spatial_features_2d"]
            else:
                key = batch_dict_k

        
        
        if data_dict['selected_char'] == 'k':
            key = self.compressor_k(key) 
            specific_prompt_k = self.specific_prompt_k.expand(key.size(0),-1,-1,-1)
            specific_prompt_k = self.compressor_k.conv_downsampling(specific_prompt_k)
            # specific_prompt_k = self.compressor_k(specific_prompt_k)

        elif data_dict['selected_char'] == 'v':
            key = self.compressor_v(key) #'v':
            specific_prompt_v = self.specific_prompt_v.expand(key.size(0),-1,-1,-1)
            specific_prompt_v = self.compressor_v.conv_downsampling(specific_prompt_v)
            # specific_prompt_v = self.compressor_v(specific_prompt_v)
        
        
        ## 将key的特征投射到自车坐标下：
        i = 0
        projected_key = []
        
        for b, cav_num in enumerate(record_len):
            neb_num = cav_num - 1
            neb_num = 1 if neb_num == 0 else neb_num # 邻车数量为0的时候补充了自车数据进去
            temp =  self.proj_neb2ego(key[i:i+neb_num], ego2neb_t_matrix[b])
            i += neb_num
            projected_key.append(temp)
        key = torch.cat(projected_key)
        
        
        # channel
        general_prompt_k = self.general_prompt_k.expand(key.size(0),-1,-1,-1)
        if data_dict['selected_char'] == 'k':
            prompts = torch.cat([specific_prompt_k, general_prompt_k],dim=1)
            key = self.channel_cross_attention(prompts, query, key, 'k')
            specific_prompt = key[:,:specific_prompt_k.size(1),:,:]
            general_prompt = key[:,specific_prompt_k.size(1):,:,:]
        elif data_dict['selected_char'] == 'v':
            prompts = torch.cat([specific_prompt_v, general_prompt_k],dim=1)
            key = self.channel_cross_attention(prompts, query, key, 'v')
            specific_prompt = key[:,:specific_prompt_v.size(1),:,:]
            general_prompt = key[:,specific_prompt_v.size(1):,:,:]


        # spatial
        specific_prompt_out = self.transformer((specific_prompt+general_prompt)/2, query)  

        

        # for style loss
        mean_k = torch.mean(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False)
        mean_k2 = torch.mean(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False)
        mean_k3 = torch.mean(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False)
        mean_q = torch.mean(query.contiguous().view(key.size(0), -1), dim=1, keepdim=False)
        
        std_k = torch.var(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_k2 = torch.var(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_k3 = torch.var(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
        std_q = torch.var(query.contiguous().view(query.size(0), -1), dim=1, keepdim=False, unbiased=False)
        
        
        # for adv loss
        out_q = self.domain_classifier(query) #[2,1,50,176]
        out_k = self.domain_classifier(general_prompt)
        
        
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

        
        if self.fusion_method=='where2comm':
            # not tested
            psm_single = self.detect_head.cls_head(spatial_features_2d)
            out, communication_rates = self.fusion_net(spatial_features_2d,
                                                                    psm_single,
                                                                    record_len,
                                                                    pairwise_t_matrix)

        if self.fusion_method=='cobevt':
            identical_matrix = torch.tile(torch.eye(4), (ego2neb_t_matrix.shape[0], 5, 5, 1, 1))
            identical_matrix = normalize_pairwise_tfm(identical_matrix, self.H, self.W, self.fake_voxel_size)
            
            out = self.fusion_net(spatial_features_2d, record_len, identical_matrix)

        # collab detection
        output_dict = self.detect_head(out)
        
        # single detection
        key_dict = self.detect_head(specific_prompt)
        
        
        # update output dict
        output_dict.update({"out_q":out_q, "out_k":out_k, "selected_char":data_dict['selected_char']})
        output_dict.update({"mean_q":mean_q, "mean_k":mean_k, "std_q":std_q, "std_k":std_k})
        output_dict.update({"mean_k2": mean_k2, "mean_k3": mean_k3, "std_k2": std_k2, "std_k3":std_k3})
        output_dict['psm_key'] = key_dict['psm']
        output_dict['rm_key'] = key_dict['rm']
        output_dict['record_len'] = record_len

        return output_dict