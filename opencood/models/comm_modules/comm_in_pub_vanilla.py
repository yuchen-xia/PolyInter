import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from opencood.models.fuse_modules.fusion_in_one import regroup
# from opencood.models.heter_pyramid_nego import HeterPyramidNego
# from opencood.models.heter_pyramid_consis import HeterPyramidConsis
# from opencood.models.heter_pyramid_consis import coords_bev_pos_emdbed
from opencood.models.sub_modules.keyfeat_modules import KeyfeatAlignPerdim, Gate2PubSemantic
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.resize_net import ResizeNet
import torchvision
from opencood.models.fuse_modules.wg_fusion_modules import Converter
from opencood.tools.feature_show import feature_show

class ComminPubVanilla(nn.Module):
    def __init__(self, args, pub_codebook=None, init_codebook=None):
        super(ComminPubVanilla, self).__init__()
        
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
        
        self.convert_phase = 'bf_decom'
        if 'convert_phase' in args.keys():
            self.convert_phase = args['convert_phase']
        
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
        
        self.H_size, self.W_size = (
            int((self.local_range[4] - self.local_range[1]) /  args_unify['granularity_H']), 
            int((self.local_range[3] - self.local_range[0]) /  args_unify['granularity_W'])
            )
        
        
        self.enhancer_send = Converter(args['converter'])
        self.aligner_send = Converter(args['converter'])

        self.aligner_receive = Converter(args['converter'])    
        self.enhancer_receive = Converter(args['converter'])

        
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
        
        # Recombine feature size to make it easy to decompose
        feature = self.recombiner_send(feature)     
        
        if not self.inference:
            self.fc_af_recombine_send = feature
        
        # if self.convert_phase == 'bf_decom':
        #     feature = self.converter_send(feature)
        
        feature = self.enhancer_send(feature)

        if not self.inference:
            self.fc_af_enhance_send = feature

        feature = self.aligner_send(feature)
            
        return feature
    
    def receiver(self, feature, record_len=None, affine_matrix=None,  record_len_modality = None):
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

        feature = self.aligner_receive(feature)

        if not self.inference:
            self.fc_bf_enhance_receive = feature

        feature = self.enhancer_receive(feature)

        if not self.inference:
            self.fc_bf_recombine_receive = feature

        feature = self.recombiner_receive(feature)

        feature = self.unify2local(feature, self.local_size)

        return feature 