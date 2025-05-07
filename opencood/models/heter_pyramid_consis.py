""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.heter_nego import HeterNego
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.comm_modules.comm_in_pub import ComminPub, coords_bev_pos_emdbed
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.tools.feature_show import feature_show
import importlib
import torchvision

def add_clone_modality_dict(output_dict, key, modality_dict):
    # 向output_dict中添加modality_dict的拷贝信息, 避免后续操作改变value值对计算loss的影响
    new_dict = {}
    for k, v in modality_dict.items():
        new_dict.update({k: v.clone()})
        
    output_dict.update({key: new_dict})

class HeterPyramidConsis(HeterNego):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        
        self.fixed = {}

        self.cosim_the = args['pub']['cor_cosim_threhold'] 
        self.local2pub_builed = False
        self.init_pub_codebook = None
        self.pub_codebook = None
        self.use_alliance = args['pub']['use_alliance']
        self.pub_cb_path = args['pub']['cb_path']
        
        self.pub_query_emb_path = args['pub']['query_emb_path']
        self.gra_H = args['pub']['granularity_H']
        self.gra_W = args['pub']['granularity_W']
        
        self.H_size, self.W_size = (
                    int((self.cav_range[4] - self.cav_range[1]) /  self.gra_H), 
                    int((self.cav_range[3] - self.cav_range[0]) /  self.gra_W)
                    )
        
        self.max_codes = 32
        self.bev_posemb = coords_bev_pos_emdbed(self.H_size, self.W_size, self.max_codes // 2)
        self.pub_query_embeddings = nn.ModuleList()
        for _ in range(self.max_codes):
            self.pub_query_embeddings.append(nn.Sequential(
            nn.Linear(self.max_codes, self.max_codes // 2),
            nn.Linear(self.max_codes // 2, 1),
            nn.ReLU(inplace=True),
            nn.Linear(1, 1)
        ))

        # if self.use_alliance:
        #     self.pub_codebook = torch.load(self.pub_cb_path)
        #     self.pub_codebook.requires_grad = False
            

        
        if args['mode_cb_same_init']:
            """为不同模态的codebook初始化相同的参数"""
            max_code_num = 0
            for modality_name in self.modality_name_list:
                code_num = args[modality_name]['comm_args']['num_codes']
                if code_num > max_code_num:
                    max_code_num = code_num
            dim_code = args['pub']['C_uni']
            init_codebook = nn.Parameter(torch.randn(max_code_num, dim_code))
        else:
            init_codebook = None  

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            
            """
            若在联盟中, 固定comm模块参数; 若不在联盟中, 调整comm模块参数
            """
            self.fixed[modality_name] = model_setting['allied']

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

            """
            Communication module from codebook
            """
            model_setting['comm_args'].update({'local_range': self.cav_range,
                                               'fixed':self.fixed[modality_name],
                                               'comm_space': args['comm_space'],
                                            #    'mode': modality_name
                                            #    'modality_name': modality_name
                                               })
            setattr(self, f"comm_{modality_name}", ComminPub(model_setting['comm_args'], self.pub_codebook, init_codebook))
            
            
            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"pyramid_backbone_{modality_name}", PyramidFusion(model_setting['fusion_backbone']))


            """
            Shrink header
            """
            setattr(self, f"shrink_flag_{modality_name}", False)
            if 'shrink_header' in model_setting:
                setattr(self, f"shrink_flag_{modality_name}", True)
                setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            """
            Shared Heads
            """
            setattr(self, f"cls_head_{modality_name}", nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(model_setting['in_head'], \
                            model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                    kernel_size=1))
        
        # compressor will be only trainable
        self.compress = False
        # if 'compressor' in args:
        #     self.compress = True
        #     self.compressor = NaiveCompressor(args['compressor']['input_dim'],
        #                                       args['compressor']['compress_ratio'])
        
        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
    
    
        """设置仅更新comm模块的参数"""    
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()
    
        for modality_name in self.modality_name_list:
            if not self.fixed[modality_name]:
                super().set_trainable(modality_name)
        # check again which module is not fixed.
        check_trainable_module(self)
        

    def to(self, device):
        super(HeterPyramidNsd, self).to(device)
        self.bev_posemb = self.bev_posemb.to(device)
    # def set_trainable(self, modality_name):
    #     for name, p in self.named_parameters():
    #         if f'comm_{modality_name}' in name:
    #             p.requires_grad = True
    #     for name, module in self.named_modules():
    #         if f'comm_{modality_name}' in name:
    #              module.train()

    # @property

    def update_pub_codebook(self):
        len_pub = self.pub_codebook.shape[0]
        
        new_pub_cb = []
        
        for pub_idx in range(len_pub):
            ref_modes = self.pub_ref_modes[pub_idx]
            
            new_pub = []
            is_new = True
            for mode in ref_modes:
                is_new = is_new and (not self.fixed[mode])
                if not is_new:
                    break
                mode_idx = self.pub2local_dict[mode][pub_idx]
                new_pub.append(self.codebook_dict[mode][mode_idx])

            if is_new:
                new_pub = torch.mean(torch.stack(new_pub), dim=0, keepdim=False)                                   
            else:
                new_pub = self.pub_codebook[pub_idx]
            
            new_pub_cb.append(new_pub)
            
        return torch.stack(new_pub_cb)
    
    @property
    def query_pub(self):
        query = []
        for c_idx in range(self.max_codes):
            if c_idx > self.pub_codebook.shape[0]:
                    break
            dim_query = self.pub_query_embeddings[c_idx](self.bev_posemb) # H*W, 1
            dim_query = dim_query.transpose(1, 0).view(1, self.H_size, self.W_size)
            query.append(dim_query)
        query = torch.cat(query, dim=0)
        return query

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single', 'modality_name_list': self.modality_name_list, 'fixed_mode':self.fixed}
        
        """Setup public code and find mapping dict from local to pub"""
        if not self.local2pub_builed:
            super().setup_pubcb_local2pub()
            self.local2pub_builed = True
            
            # 设置的未加入联盟的qurey_embeddings参数可训练
            # acitive_code_idxs = self.local2pub_dict
            for c_idx in range(self.max_codes):     
                if c_idx > self.pub_codebook.shape[0]:
                    break
                           
                ref_modes = self.pub_ref_modes[c_idx]
                update_query_dim = True
                for mode in ref_modes: # 只要该维度对应的模态中, 有已经加入联盟的, 就无需训练
                    if self.fixed[mode]:
                        update_query_dim = False
                        break
                if update_query_dim:
                    self.pub_query_embeddings[c_idx].requires_grad = True
                    
        # print(self.pub_query_embeddings[0][0].weight)
        # print(self.pub_query_embeddings[10][0].weight)
        # 将pub与local的关联信息传递到本地
        codebook_dict = {}
        for modality_name in self.modality_name_list:
            eval(f'self.comm_{modality_name}.pub_change')(self.local2pub_dict[modality_name], \
                                                    self.pub2local_dict[modality_name], self.pub_codebook)
            if not self.fixed[modality_name]:
                codebook_dict[modality_name] = eval(f'self.comm_{modality_name}.codebook')
            self.codebook_dict = codebook_dict
        output_dict.update({"codebook_dict": codebook_dict, "local2pub_dict":self.local2pub_dict})  
        
        
        self.pub_codebook = self.update_pub_codebook()
        output_dict.update({"pub_codebook": self.pub_codebook}) 
        
        # modality_type_list = data_dict['modality_type_list'] 
        # agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_feature_dict = {}
        
        for modality_name in self.modality_name_list:
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
            
        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if self.sensor_type_dict[modality_name] == "camera":
                # should be padding. Instead of masking
                feature = modality_feature_dict[modality_name]
                _, _, H, W = feature.shape
                target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                # 对齐特征尺寸
                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                modality_feature_dict[modality_name] = crop_func(feature)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict.update({
                        f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                    })
        
        
        add_clone_modality_dict(output_dict, "fc_before_send", modality_feature_dict)                
        
        mode_pub_weight_dict = {}
        fc_before_rbdc = {}
        fc_after_rbdc = {}
        for modality_name in self.modality_name_list:
            mode_pub_weight_dict[modality_name] = \
              eval(f"self.comm_{modality_name}.sender")(modality_feature_dict[modality_name], self.query_pub)
            
            if not self.fixed[modality_name]:
                fc_before_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_before_rbdc")})    
                fc_after_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_after_rbdc")})
            
            # ref_pub_idxs[modality_name] = (eval(f"self.comm_{modality_name}.ref_pub_idxs"))
          
        # 计算公共向量的平均值
        weight_pub = torch.sum(torch.stack(list(mode_pub_weight_dict.values())), dim=0)
        weight_pub = weight_pub / self.ref_num_modes_per_pubcode.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # feature_show(weight_pub[0], f'analysis/feature_maps/w_pub_avg')
        # torch.max(weight_pub)
        
        add_clone_modality_dict(output_dict, "mode_pub_weight_dict", mode_pub_weight_dict)
        output_dict.update({"weigh_pub": weight_pub.clone()})
        
        shared_weights_bf_trans = {}
        shared_weights_af_trans = {}
        for modality_name in self.modality_name_list:
            modality_feature_dict[modality_name] = \
                eval(f"self.comm_{modality_name}.receiver")(weight_pub, record_len, affine_matrix)
            
            if not self.fixed[modality_name]:
                shared_weights_bf_trans.update({modality_name: eval(f"self.comm_{modality_name}.shared_weights_bf_trans")})
                shared_weights_af_trans.update({modality_name: eval(f"self.comm_{modality_name}.shared_weights_af_trans")})
        
        add_clone_modality_dict(output_dict, "fc_after_send", modality_feature_dict)
        
        output_dict.update({
            "fc_before_rbdc": fc_before_rbdc,
            "fc_after_rbdc": fc_after_rbdc,
            "shared_weights_bf_trans": shared_weights_bf_trans,
            "shared_weights_af_trans": shared_weights_af_trans,
            
        })
        return output_dict