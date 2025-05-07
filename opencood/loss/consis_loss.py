import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.loss.point_pillar_loss import sigmoid_focal_loss

class ConsisLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super().__init__(args)
        
        # # 默认不使用pub_codebook监督
        # self.pub_cb_supervise = False
        # if 'pub_cb_supervise' in args:
        #     self.pub_cb_supervise = args['pub_cb_supervise']
  
        #     pub_cb_path = 'opencood/logs/pub_codebook/pub_codebook.pth'
        #     if 'pub_cb_path' in args:
        #         pub_cb_path = args['pub_cb_path']
        #     self.codebook_pub = torch.load(pub_cb_path)
        #     # print(torch.norm(self.codebook_pub, dim=1))
            
        # self.unit_loss = True
        # if 'unit_loss' in args:
        #     self.unit_loss = args['unit_loss']
    
            
        # self.cosim_threhold = args['cosim_threhold']
        
        # self.num_unmatched_code = 0

        self.mse_loss = nn.MSELoss()
        
        self.loss_dict = {}
    
    def forward(self, output_dict, target_dict, suffix=""):
        """
        output_dict:{
            'modality_name_list': modality names
            "pub_codebook": public codebook
            'codebook_dict': meta sematics for each modality
            'pub_weight_dict': pub weights per modality
            'weigh_pub': average of pub weights 
            'local2pub_dict' local to pub mapping dict for each modality
            'fc_before_send': feature before send
            'fc_after_send': feature after send
            'fc_before_rbdc': feature before reconstruct by decomposed codebooks,
            'fc_after_rbdc': feature before reconstruct by decomposed codebooks,
            'shared_weights_bf_trans'
            "shared_weights_af_trans"
            
        }
        """
        
        self.modality_name_list = output_dict['modality_name_list']
        pub_codebook = output_dict['pub_codebook']
        codebook_dict = output_dict['codebook_dict']
        mode_pub_weight_dict = output_dict['mode_pub_weight_dict']
        weight_pub = output_dict['weigh_pub']
        local2pub_dict = output_dict['local2pub_dict']
        
        fc_before_send = output_dict['fc_before_send']
        fc_after_send = output_dict['fc_after_send']
        fc_before_rbdc = output_dict['fc_before_rbdc']
        fc_after_rbdc = output_dict['fc_after_rbdc']        
        shared_weights_bf_trans = output_dict['shared_weights_bf_trans']
        shared_weights_af_trans = output_dict['shared_weights_af_trans']
        
        total_loss = 0
        
        rec_loss = 0
        unit_loss = 0
        orth_loss = 0
        cycle_loss = 0
        
        cb_unify_loss = 0
        w_unify_loss = 0
        w_cycle_loss = 0
        
        modality_loss_dict = {}
        for modality_name in self.modality_name_list:
            m_rec_loss = self.mse_loss(fc_before_rbdc[modality_name], fc_after_rbdc[modality_name])
            
            
            m_codebook = codebook_dict[modality_name]
            m_cb_len = m_codebook.shape[0]
            m_unit_loss = self.mse_loss(torch.norm(m_codebook, p=2, dim=1),
                                        torch.ones((m_cb_len)).to(m_codebook.device))
            
            ref_local_idx = []
            ref_pub_idx = []
            for idx_local in range(m_cb_len):
                if idx_local in local2pub_dict[modality_name].keys():
                    idx_pub = local2pub_dict[modality_name][idx_local]
                    
                    ref_pub_idx.append(idx_pub)
                    ref_local_idx.append(idx_local)
            m_cb_unify_loss = 1 - F.cosine_similarity(m_codebook[ref_local_idx], pub_codebook[ref_pub_idx], dim=1)
            m_cb_unify_loss = torch.sum(m_cb_unify_loss)
            
            m_codebook = F.normalize(m_codebook, p=2, dim=1)
            m_cosim_mat = torch.mm(m_codebook, m_codebook.t())
            identity_matrix = torch.eye(m_cb_len).to(m_codebook.device)
            m_orth_loss = self.mse_loss(m_cosim_mat, identity_matrix)
            
            m_cycle_loss = self.mse_loss(fc_before_send[modality_name], fc_after_send[modality_name])
            
            m_w_unify_loss = self.mse_loss(mode_pub_weight_dict[modality_name][:, ref_pub_idx], \
                                                weight_pub[:, ref_pub_idx])
            
            m_w_cycle_loss = self.mse_loss(shared_weights_af_trans[modality_name], shared_weights_bf_trans[modality_name])
            
            rec_loss = rec_loss + m_rec_loss 
            unit_loss = unit_loss + m_unit_loss 
            orth_loss = orth_loss + m_orth_loss 
            cycle_loss = cycle_loss + m_cycle_loss 
            cb_unify_loss = cb_unify_loss + m_cb_unify_loss 
            w_unify_loss = w_unify_loss + m_w_unify_loss 
            w_cycle_loss = w_cycle_loss + m_w_cycle_loss 
            
            modality_loss = m_rec_loss + m_unit_loss + m_orth_loss + m_cycle_loss + \
                            m_cb_unify_loss + m_w_unify_loss + m_w_cycle_loss 
                            
            modality_loss_dict[modality_name] = modality_loss
            total_loss = total_loss + modality_loss
        
        self.total_loss = total_loss
        self.loss_dict.update({
            "rec_loss": rec_loss,
            "unit_loss": unit_loss,
            "orth_loss": orth_loss,
            "cycle_loss": cycle_loss,
            "cb_unify_loss ": cb_unify_loss ,
            "w_unify_loss": w_unify_loss,
            "w_cycle_loss": w_cycle_loss,
            
        })
       
        return total_loss
        
    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        
        for k, v in self.loss_dict.items():
            writer.add_scalar(k, v.item(), epoch*batch_len + batch_id)
        
        writer.add_scalar('Total_loss', self.total_loss.item(),
                            epoch*batch_len + batch_id)
           
        print_msg ="[epoch %d][%d/%d], || Loss: %.4f" % (epoch, batch_id + 1, batch_len, self.total_loss.item())
        for k, v in self.loss_dict.items():
            k = k.replace("_loss", "").capitalize()
            print_msg = print_msg + f" || {k}: {v:.4f}"
        
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)
    
    
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
