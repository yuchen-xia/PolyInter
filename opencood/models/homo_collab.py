""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
# from opencood.models.sub_modules.comm_in_local import ComminLocal
from opencood.models.comm_modules.comm_in_pub import ComminPub
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.tools.feature_show import feature_show
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision

class HomoCollab(nn.Module):
    def __init__(self, args):
        super(HomoCollab, self).__init__()
        self.args = args
        
        modality_name = args['ego_modality']
        self.modality_name = modality_name
        fusion_name = args[modality_name]['fusion_net']['method']

        self.cav_range = args['lidar_range']

        self.cam_crop_info = {} 
        
        
        # setup each modality model
        # for modality_name in self.modality_name_list:
        model_setting = args[modality_name]
        sensor_name = model_setting['sensor_type']
        self.sensor_type = model_setting['sensor_type']

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
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        setattr(self, f"fusion_net_{modality_name}", build_fusion_net(model_setting['fusion_net']))
        setattr(self, f"fusion_method_{modality_name}", fusion_name)


        """
        Shrink header
        """
        if fusion_name == 'pyramid':
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

            
    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] # 场景内智能车数量
        # print(record_len)
        # print(agent_modality_list)
        # modality_count_dict = Counter(agent_modality_list)
        # modality_feature_dict = {}

        # 按modality分类推理
        # for modality_name in self.modality_name_list:
            # if modality_name not in modality_count_dict:
                # continue
        modality_name = self.modality_name
        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)
        # modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        # for modality_name in self.modality_name_list:
        #     if modality_name in modality_count_dict:
        if self.sensor_type == "camera":
            # should be padding. Instead of masking
            # feature = modality_feature_dict[modality_name]
            _, _, H, W = feature.shape
            target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
            target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

            # 对齐特征尺寸, 使特征表示的范围相同
            crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
            feature = crop_func(feature)
            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })        

        """Decode and output"""
        fusion_method = eval(f"self.fusion_method_{modality_name}")
        if fusion_method == 'pyramid':
            fused_feature, occ_outputs = eval(f"self.fusion_net_{modality_name}.forward_collab")(
                                                    feature,
                                                    record_len, 
                                                    affine_matrix, 
                                                    agent_modality_list, 
                                                    self.cam_crop_info,
                                                )
        else:
            fused_feature = eval(f'self.fusion_net_{modality_name}')(feature, record_len, affine_matrix)
            occ_outputs = None


        if fusion_method == 'pyramid' and eval(f"self.shrink_flag_{modality_name}"):
            fused_feature = eval(f"self.shrink_conv_{modality_name}")(fused_feature)

        cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
        reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
        dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        return output_dict
