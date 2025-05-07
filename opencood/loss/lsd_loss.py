import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.loss.point_pillar_loss import sigmoid_focal_loss

class LSDLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super().__init__(args)
        
        # 默认不使用pub_codebook监督
        self.pub_cb_supervise = False
        if 'pub_cb_supervise' in args:
            self.pub_cb_supervise = args['pub_cb_supervise']
  
            pub_cb_path = 'opencood/logs/pub_codebook/pub_codebook.pth'
            if 'pub_cb_path' in args:
                pub_cb_path = args['pub_cb_path']
            self.codebook_pub = torch.load(pub_cb_path)
            # print(torch.norm(self.codebook_pub, dim=1))
            
        self.unit_loss = True
        if 'unit_loss' in args:
            self.unit_loss = args['unit_loss']
    
            
        self.cosim_threhold = args['cosim_threhold']
        
        self.num_unmatched_code = 0

        self.mse_loss = nn.MSELoss()
        
        self.loss_dict = {}

    def calc_sg_loss(self, sg_map, target_dict):
        
        batch_size = target_dict['pos_equal_one'].shape[0]
        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
        pos_normalizer = positives.sum(1, keepdim=True).float()
        
        
        cls_preds = sg_map.permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size
        
        
        return cls_loss
    
    def forward(self, output_dict, target_dict, suffix=""):
        """
        output_dict:{
            'sg_map': significant map of feature after recombined
            'fc_before_send': feature before send
            'fc_after_send': feature after send
            'fc_before_rbdc': feature before reconstruct by decomposed codebooks,
            'fc_after_rbdc': feature before reconstruct by decomposed codebooks,
            'codebook': decomposed sematics
            
        }
        """
        # directory = 'opencood/logs/HeterBaseline_opv2v_lidar_m1-v3/comm_module/analysis'
        
        # feature_show(output_dict['fc_before_send'][0], os.path.join(directory, 'fc_before_send.png'))
        # feature_show(output_dict['fc_after_send'][0], os.path.join(directory, 'fc_after_send.png'))
        # feature_show(output_dict['fc_before_rbdc'][0], os.path.join(directory, 'fc_before_rbdc.png'))
        # feature_show(output_dict['fc_after_rbdc'][0], os.path.join(directory, 'fc_after_rbdc.png'))
        
        # loss of significant seletor 
        # sg_loss = self.calc_sg_loss(output_dict['sg_map'], target_dict)
        sg_loss = 0
        
        # reg , cls, occ_loss
        # pyramid_loss = super().forward(output_dict, target_dict, suffix)
        # self.loss_dict["reg_loss"] = 0
        # self.loss_dict["cls_loss"] = 0
        
        cycle_loss = self.mse_loss(output_dict['fc_before_send'], output_dict['fc_after_send'])
        rec_loss = self.mse_loss(output_dict['fc_before_rbdc'], output_dict['fc_after_rbdc'])
        
        
        
        codebook = output_dict['codebook']
        num_codes = codebook.size()[0]
        
        
        # gram_matrix = torch.matmul(codebook, codebook.t())
        # corr_matrix = gram_matrix
        
        # 单位向量损失
        if self.unit_loss:
            unit_loss = self.mse_loss(torch.norm(codebook, p=2, dim=1),
                                torch.ones((num_codes)).to(codebook.device))
        else: 
            unit_loss = torch.Tensor([0]).squeeze().to(codebook.device)
            unit_loss.requires_grad = False

        norm_codebook = torch.norm(codebook, p=2, dim=1, keepdim=True)
        codebook = codebook / norm_codebook
        cosim_matrix = torch.matmul(codebook, codebook.t())
        corr_matrix = cosim_matrix
        
        identity_matrix = torch.eye(num_codes, device=codebook.device)
        
        # 正交损失
        orth_loss = self.mse_loss(corr_matrix, identity_matrix)
        
        # total_loss = rec_loss + cycle_loss + orth_loss
        # total_loss = rec_loss + cycle_loss + orth_loss + unit_loss + det_loss
        # total_loss = rec_loss + cycle_loss + orth_loss + unit_loss + sg_loss
        total_loss = rec_loss + cycle_loss + orth_loss + unit_loss
        
        
        """
        从local codebook 和 public codebook中找到最匹配的基向量, 并最小化余弦相似度使其靠近
        local codebook中未匹配的向量加入public codebook
        """
        self.loss_dict['unify_loss'] = torch.Tensor([0]).squeeze().to(codebook.device)
        if self.pub_cb_supervise:
            unify_loss =  torch.Tensor([0]).squeeze().to(codebook.device)
            codebook_pub = self.codebook_pub
            codebook_pub.requires_grad = False
            len_local = codebook.size()[0]
            len_pub = codebook_pub.size()[0]
            

            # 计算local和public的codebook余弦相似度矩阵
            norm_local = F.normalize(codebook, p=2, dim=1)
            norm_pub = F.normalize(codebook_pub, p=2, dim=1)
            cosim_mat = torch.mm(norm_local, norm_pub.t()) # (len_local, len_pub)
            
            
            # 本地到pub码本的映射关系：[0, 1, ...] -> [pub_idx1, pub_idx2, ...]
            map_to_pub = -1 * torch.ones(len_local).to(torch.int64).to(codebook.device)
            while len_local>0 and len_pub > 0:
                # 找到余弦相似度最接近1的idx_local和idx_pub
                cosim_max_per_local, indicies_pub = torch.max(cosim_mat, dim=1)
                cosim_max, idx_local = torch.max(cosim_max_per_local.unsqueeze(0), dim=1)
                
                # 余弦相似度大于阈值才视为相关                
                # 若矩阵中最大余弦相似度仍小于阈值, 退出循环
                if cosim_max < self.cosim_threhold:
                    break
                
                idx_pub = indicies_pub[idx_local] # idx_local既是匹配到的本地基向量对应的索引, 也可以找到公共基向量的索引
                map_to_pub[idx_local] = idx_pub
                # unify_loss = unify_loss + (1 - cosim_min.detach()).squeeze()
                unify_loss = unify_loss + (1 - torch.dot(norm_local[idx_local][0], norm_pub[idx_pub][0]))
                
                # idx_local对应的行, idx_pub对应的列置为inf, 不再参与匹配
                # cosim_mat[idx_local,:] = float('inf')
                # cosim_mat[:,idx_pub] = float('inf')
                cosim_mat[idx_local,:] = 0
                cosim_mat[:,idx_pub] = 0
                len_local = len_local - 1
                len_pub = len_pub - 1
            
            self.num_unmatched_code = len_local 
            
            # 若local codebook中还有未参与匹配的code, 加入pub_codebook    
            if len_local > 0:
                cosim_min, _ = torch.min(cosim_mat, dim=1)
                idx_newcode = torch.nonzero(~torch.isinf(cosim_min)).squeeze(1)
                codebook_pub = torch.cat((codebook_pub, codebook[idx_newcode,:]), dim=0)
                self.final_codebook_pub = codebook_pub
                
                # 为剩余的local codebook 赋予映射编号
                len_pub = codebook_pub.shape[0]
                map_to_pub[idx_newcode] = torch.arange(len_pub, len_pub + len_local).to(codebook.device)
                self.map_to_pub = map_to_pub  
                # torch.save(codebook_pub, 'opencood/logs/pub_codebook/pub_codebook.pth')
            total_loss = total_loss + unify_loss
            
            self.loss_dict.update({"unify_loss": unify_loss})
                                        
            
        
        self.loss_dict.update({
            "rec_loss": rec_loss,
            "cycle_loss": cycle_loss,
            'orth_loss': orth_loss,
            'unit_loss': unit_loss,
            # 'sg_loss': sg_loss,
            'total_loss': total_loss
        })
        
        return total_loss
        
    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        # self.loss_dict["reg_loss"] = 0
        # self.loss_dict["cls_loss"] = 0
        rec_loss = self.loss_dict["rec_loss"]
        cycle_loss = self.loss_dict["cycle_loss"]
        orth_loss = self.loss_dict["orth_loss"]
        unit_loss =  self.loss_dict["unit_loss"]
        unify_loss =  self.loss_dict["unify_loss"]
        # reg_loss =  self.loss_dict["reg_loss"]
        # cls_loss =  self.loss_dict["cls_loss"]
        # sg_loss = self.loss_dict['sg_loss']
        total_loss = self.loss_dict["total_loss"]
        
        if writer is not None:
            writer.add_scalar('rec_loss', rec_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('cycle_loss', cycle_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Orth_loss', orth_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Unit_loss', unit_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Unify_loss', unify_loss.item(),
                            epoch*batch_len + batch_id)
            # writer.add_scalar('Reg_loss', reg_loss,
            #                 epoch*batch_len + batch_id)
            # writer.add_scalar('Cls_loss', cls_loss,
            #                 epoch*batch_len + batch_id)      
            # writer.add_scalar('Sg_loss', sg_loss,
            #                 epoch*batch_len + batch_id)       
            writer.add_scalar('Total_loss', total_loss.item(),
                            epoch*batch_len + batch_id)

        print_msg ="[epoch %d][%d/%d], || Loss: %.4f || Cycle: %.4f || Rec: %.4f || Orth: %.4f || Unit: %.4f || Unfy: %.4f" % \
            (
                epoch, batch_id + 1, batch_len,
                total_loss.item(), orth_loss.item(), rec_loss.item(), \
                cycle_loss.item(), unit_loss.item(), unify_loss.item()
            )
            
        # print_msg ="[epoch %d][%d/%d], || Loss: %.4f || Orth: %.4f || Rec: %.4f || Cycle: %.4f || Unit: %.4f|| Cls: %.4f|| Reg: %.4f|| Sg: %.4f" % \
        #     (
        #         epoch, batch_id + 1, batch_len,
        #         total_loss.item(), orth_loss.item(), rec_loss.item(), \
        #         cycle_loss.item(), unit_loss.item(), cls_loss, reg_loss, sg_loss
        #     )
        # print(print_msg)
        # pbar = None
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)
