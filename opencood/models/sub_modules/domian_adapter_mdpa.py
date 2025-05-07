import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.pyramid_fuse import regroup, warp_affine_simple
from opencood.models.fuse_modules.wg_fusion_modules import CrossDomainFusionEncoder, SwapFusionEncoder


class residual_block(nn.Module):
    def __init__(self, input_dim):
        super(residual_block, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = x + self.module(x)
        return x


class LearnableResizer(nn.Module):
    def __init__(self, args):
        super(LearnableResizer, self).__init__()
        # channel selection
        self.channel_selector = nn.Conv2d(args['input_channel'],
                                          args['output_channel'],
                                          1)
        # window+grid attention
        self.wg_att_1 = SwapFusionEncoder(args['wg_att'])
        self.wg_att_2 = SwapFusionEncoder(args['wg_att'])


        self.res_blocks = nn.ModuleList()
        num_res = args['residual']['depth']
        input_channel = args['residual']['input_dim']

        # residual block
        for i in range(num_res):
            self.res_blocks.append(residual_block(input_channel))

    def forward(self, ego_feature, cav_feature):
        cav_feature = self.channel_selector(cav_feature)

        _, h, w = ego_feature.shape
        # self attention
        cav_feature_1 = self.wg_att_1(cav_feature)
        # naive feature resizer
        cav_feature_1 = torch.nn.functional.interpolate(cav_feature_1,
                                                      [h,
                                                       w],
                                                      mode='bilinear',
                                                      align_corners=False)
        cav_feature_2 = cav_feature_1
        for res_bloc in self.res_blocks:
            cav_feature_2 = res_bloc(cav_feature_2)
        cav_feature_2 += cav_feature_1
        cav_feature_2 = self.wg_att_2(cav_feature_2)

        # residual shortcut
        cav_feature_0 = torch.nn.functional.interpolate(cav_feature,
                                                      [h,
                                                       w],
                                                      mode='bilinear',
                                                      align_corners=False)
        return cav_feature_0 + cav_feature_2

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        inner_dims = in_channels * 2

        self.conv1_da = nn.Conv2d(in_channels, inner_dims, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(inner_dims, 1, kernel_size=1, stride=1)
        self.rgl = GradientScalarLayer(-9.1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # X : (b, c, h, w), b=cars_1 + cars_2 + ...
        x = self.rgl(x)
        x = F.relu(self.conv1_da(x))
        x = self.conv2_da(x)

        return x


class DomianAdapterMdpa(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.local_modality = args['local_modality']
        
        share_feat_dim = args['share_feat_dim']
        self.local_dim = share_feat_dim[self.local_modality]

        modality_name_list = ['m1', 'm2', 'm3', 'm4']
        neb_modality_name_list = list(set(modality_name_list) - set(self.local_modality))
        self.neb_modality_name_list = neb_modality_name_list
        for modality_name in neb_modality_name_list:
            neb_dim = share_feat_dim[modality_name]
            args['resizer'].update({'input_channel': neb_dim,
                                    'output_channel': self.local_dim,
                                    })
            args['resizer']['wg_att'].update({'input_dim': neb_dim,
                                              'mlp_dim': neb_dim })
            args['resizer']['residual'].update({'input_dim': neb_dim})
            setattr(self, f'resizer_for_{modality_name}', LearnableResizer(args['resizer']))
            
            
            args['cdt'].update({'input_dim': neb_dim})
            setattr(self, f'cdt_for_{modality_name}', CrossDomainFusionEncoder(args['cdt']))
            
            setattr(self, f'classifier_for_{modality_name}', DAImgHead(self.local_dim))
        # self.classifier = DAImgHead(self.local_dim)
            

    def forward(self, x, agent_modality_list, record_len, affine_matrix):
        """neb feat convert"""
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]
        split_x = regroup(x, record_len)
        
        mode_idx = 0
        out = []
        da_cls_pred_out = []
        source_idx_out = []
        source_idx = 0
        for b in range(B):
            N = record_len[b]
            batch_features = split_x[b]
            batch_out = []
            batch_da_cls_pred_out = []
            
            
            # 提取ego
            ego = batch_features[0].unsqueeze(0) # 1 c h  w
            # ego =  eval(f'self.enhancer_for_m2')(ego)
            batch_out.append(ego)
            mode_idx = mode_idx + 1
            
            if N > 1:
                # 提取neb, 若模态与ego不同, 转换
                # score = split_score[b]
                t_matrix = affine_matrix[b][:N, :N, :, :]
                # ego 映射到neb视角, 作为查询
                i = 0 # ego
                ego_expand = ego.expand(N-1, -1, -1, -1)
                ego_in_neb = warp_affine_simple(ego_expand,
                                                t_matrix[1:, i, :, :],
                                                (H, W))
                
                for neb_id in range(0, N-1):
                    neb_mode = agent_modality_list[mode_idx]
                    neb_feature = batch_features[neb_id + i].unsqueeze(0)
                    if neb_mode != self.local_modality:
                        neb_feature = eval(f'self.resizer_for_{neb_mode}')\
                            (ego_in_neb[neb_id], neb_feature)
                        neb_feature = eval(f'self.cdt_for_{neb_mode}')(ego_in_neb[neb_id].unsqueeze(0), neb_feature)
                        
                        
                        # 为每个neb的classifier分别加入对应的ego_in_neb辅助训练
                        da_cls_pred_ego = eval(f'self.classifier_for_{neb_mode}')(ego_in_neb[neb_id].unsqueeze(0)) 
                        # da_cls_pred_ego = eval(f'self.classifier_for_{neb_mode}')(ego) 
                        da_cls_pred_neb = eval(f'self.classifier_for_{neb_mode}')(neb_feature)                        
                        batch_da_cls_pred_out.extend([da_cls_pred_ego, da_cls_pred_neb])
                        source_idx_out.append(source_idx)
                        source_idx = source_idx + 2
                        
                    batch_out.append(neb_feature)
                    mode_idx = mode_idx + 1
                    
            
            batch_out = torch.cat(batch_out, dim=0)
            
            # 没有neb, 对batch_da_cls_pred补0
            if len(batch_da_cls_pred_out) == 0:
                 batch_da_cls_pred_out = [torch.ones((1, 1, H, W)).to(x.device)]
                 source_idx = source_idx + 1
            batch_da_cls_pred_out = torch.cat(batch_da_cls_pred_out, dim=0)
            
            out.append(batch_out)
            da_cls_pred_out.append(batch_da_cls_pred_out)
        
        out = torch.cat(out, dim=0)
        da_cls_pred_out = torch.cat(da_cls_pred_out, dim=0)
        
        
        return out, da_cls_pred_out, source_idx_out