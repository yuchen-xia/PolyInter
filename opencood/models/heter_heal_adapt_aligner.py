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
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.tools.feature_show import feature_show
import importlib
import torchvision

class HeterHealAdaptAligner(nn.Module):
    def __init__(self, args):
        super(HeterHealAdaptAligner, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        self.allied = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            
            self.allied[modality_name] = model_setting['allied']

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


            """
            Shrink header
            """
            setattr(self, f"shrink_flag_{modality_name}", False)
            if 'shrink_header' in model_setting:
                setattr(self, f"shrink_flag_{modality_name}", True)
                setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            """
            Local Heads
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

        """ Setup Shared fusion net and heads """
        shared_args = args['shared_head']

        self.fusion_name = shared_args['fusion_net']['method']
        self.shared_fusion_net = build_fusion_net(shared_args['fusion_net'])
        
        self.shrink_flag = False
        if 'shrink_header' in shared_args.keys():
            self.shared_shrink_header = DownsampleConv(shared_args['shrink_header'])
            self.shrink_flag = True
        
        self.shared_cls_head = nn.Conv2d(shared_args['in_head'], shared_args['anchor_number'], kernel_size=1
                                         )
        self.shared_ref_head = nn.Conv2d(shared_args['in_head'], 7 * shared_args['anchor_number'],
                                kernel_size=1)
        self.shared_dir_head = nn.Conv2d(shared_args['in_head'], \
                        shared_args['dir_args']['num_bins'] * shared_args['anchor_number'],
                                kernel_size=1)
            
        
        """ For feature transformation """
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        # self.model_train_init()
        # check again which module is not fixed.
        """ 预设置全部参数冻结 """         
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()


        """ Set parameters """
        self.allied_modality_list = []
        for modality_name in self.modality_name_list:
            if not self.allied[modality_name]:
                self.set_aligner_trainable(modality_name)
                print(f"New agent type: {modality_name}")
            else:
                # 不固定参数, 即已加入联盟, 记录下来
                self.allied_modality_list.append(modality_name)
                print(f"Allied agent type: {modality_name}")
                
        if len(self.allied_modality_list) == 0:
            self.set_shared_module_trainable()
        
        self.newtype_modality_list = list(set(self.modality_name_list) - set(self.allied_modality_list))
        
        
        check_trainable_module(self)


    def set_shared_module_trainable(self):
        for name, p in self.named_parameters():
            if 'shared' in name:
                p.requires_grad = True
        for name, module in self.named_modules():
            if 'shared' in name:
                module.train()    

    def set_aligner_trainable(self, modality_name):
        for name, p in self.named_parameters():
            if (modality_name in name) and ('aligner' in name):
                p.requires_grad = True
        for name, module in self.named_modules():
            if (modality_name in name) and ('aligner' in name):
                module.train()
                

    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
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

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        # feature_show(heter_feature_2d[0], '/home/scz/HEAL/analysis/m1', type = 'mean')
        
        # if self.compress:
        #     heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        
        """Fuse and output by shared fusion and head"""
        if self.fusion_name == 'pyramid':
            fused_feature, occ_outputs = self.shared_fusion_net.forward_collab(
                                                    heter_feature_2d,
                                                    record_len, 
                                                    affine_matrix, 
                                                    agent_modality_list, 
                                                    self.cam_crop_info
                                                )

            
            fused_feature = self.shared_shrink_header(fused_feature)
        else:
            fused_feature = self.shared_fusion_net(heter_feature_2d, record_len, affine_matrix)
            occ_outputs = None

        cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
        reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
        dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)

        # feature_show(cls_preds[0], 'analysis/cls_pred')

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        return output_dict
