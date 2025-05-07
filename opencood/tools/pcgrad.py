from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, model, hypes):
        # self.model = model
        self.nego_modality_list = model.newtype_modality_list
        self.para_group = hypes['train_setting']['pc_grad_group']
        
        self.modality_negotiator = {}
        for modality_name in self.nego_modality_list:
            self.modality_negotiator[modality_name] = eval(f'model.negotiator_bak_{modality_name}')
            # self.modality_negotiator[modality_name] = eval(f'model.comm_{modality_name}.negotiator_send')
            
        self.group_gard()
        
        
    def group_gard(self):
        modality_flag = self.nego_modality_list[0]
        
        self.group_name_list = []
        
        if self.para_group:
            # layer and estimators
            for layer_idx in range(0, self.modality_negotiator[modality_flag].resnet.layernum):
                self.group_name_list.append(f'layer{layer_idx}')    
            
            # deblock and output
            self.group_name_list.extend(['shrink_header', 'deblocks']) 
        else:
            self.group_name_list = ['e']
        
        # self.group_name_list = ['w', 'b']
        
        """ 
            group_grad_dict: {para_group1: {mode1: grads, mode2: grads} }
            group_shape_dict: {para_group1: {mode1: grads, mode2: grads} }
        """
        modality_dict = OrderedDict((modality_name, []) for modality_name in self.nego_modality_list)
        self.group_grad_dict = OrderedDict((group_name, copy.deepcopy(modality_dict)) for group_name in self.group_name_list)
        self.group_shape_dict = OrderedDict((group_name, copy.deepcopy(modality_dict)) for group_name in self.group_name_list)
        self.group_pname_dict = OrderedDict((group_name, copy.deepcopy(modality_dict)) for group_name in self.group_name_list)
        
        modality_flag = self.nego_modality_list[0]
        for modality_name in self.nego_modality_list:
            for name, p in self.modality_negotiator[modality_flag].named_parameters():
                # g = p.grad.clone()
                # g_shape = g.shape
                
                group_flag = False
                for group_name in self.group_name_list:
                    if group_name in name:
                        # self.group_shape_dict[group_name][modality_name].append(g_shape)
                        self.group_pname_dict[group_name][modality_name].append(name)
                        group_flag = True
                assert group_flag == True
            
            
    def grad_surgery(self):
        '''
        input:
        - objectives: a list of objectives
        '''

        self._pack_grad()
        pc_grad = self._project_conflicting()
        unflatten_grad = self._unflatten_grad(pc_grad)
        self._set_grad(unflatten_grad)
        return
        
    
    def _pack_grad(self):
        # modality_flag = self.nego_modality_list[0]
        
        for modality_name in self.nego_modality_list:
            # 重置gard
            for group_name in self.group_name_list:
                self.group_grad_dict[group_name][modality_name] = []
                self.group_shape_dict[group_name][modality_name] = []
            for name, p in self.modality_negotiator[modality_name].named_parameters():
                # p = eval(name.replace(modality_flag, modality_name))
                g = p.grad.clone()
                g_shape = g.shape
                group_flag = False
                for group_name in self.group_name_list:
                    if group_name in name:
                        self.group_grad_dict[group_name][modality_name].append(g.flatten())
                        self.group_shape_dict[group_name][modality_name].append(g_shape)
                        group_flag = True
                assert group_flag == True
                
        for group_name in self.group_name_list:
            for modality_name in self.nego_modality_list:
                self.group_grad_dict[group_name][modality_name] = torch.cat(self.group_grad_dict[group_name][modality_name])

    def _project_conflicting(self):
        """在每个参数分组, 消除跨模态提取冲突, 返回参数分组梯度平均值"""
        proj_grad_dict = copy.deepcopy(self.group_grad_dict)
        
        for group_name in self.group_name_list:
            for mode_i in self.nego_modality_list:
                g_i = proj_grad_dict[group_name][mode_i]
                for mode_j in self.nego_modality_list:
                    g_j = self.group_grad_dict[group_name][mode_j]
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0: # 原地操作, 直接对g_i进行修改
                        g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
            
            proj_grad_dict[group_name] = torch.stack(list(proj_grad_dict[group_name].values())).mean(dim=0)
            
        return proj_grad_dict

    def _unflatten_grad(self, proj_grad_dict):
        unflatten_grad_dict = OrderedDict()
        modality_flag = self.nego_modality_list[0]

        """将每个分组内参数的梯度重组回原有形状, 返回name-grad的字典"""
        for group_name in self.group_name_list:
            # unflatte_grad_dict[group_name] = []
            group_shaps = self.group_shape_dict[group_name][modality_flag]
            group_names = self.group_pname_dict[group_name]
            
            grads = proj_grad_dict[group_name]
            g_idx = 0
            # shape_idx = 0
            for shape_idx in range(len(group_shaps)):
                shape = group_shaps[shape_idx]
                tg_len = np.prod(shape)
                unflatte_grad = grads[g_idx: g_idx + tg_len].view(shape).clone()
                g_idx = g_idx + tg_len

                # 每个模态的unflatte_grad均设置为平均值, 同步加入dict
                for modality_name in self.nego_modality_list:
                    unflatten_grad_dict[group_names[modality_name][shape_idx]] = unflatte_grad
                
                
        return unflatten_grad_dict
        
  
    def _set_grad(self, unflatten_grad_dict):
        """将参数梯度的平均值分别赋给各个模态"""
        for mode_nego in self.modality_negotiator.values():
            for pname, p in mode_nego.named_parameters():
                p.grad = unflatten_grad_dict[pname]
    


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.newtype_modality_list = ['m1', 'm2']
        self.comm_m1 = nn.Linear(2, 2)
        self.comm_m2 = nn.Linear(2, 2)
        self.out_1 = nn.Linear(2, 1)
        self.out_2 = nn.Linear(2, 1)
        
        
        for name, p in self.comm_m1.named_parameters():
            print(name, p)
        for name, p in self.comm_m2.named_parameters():
            print(name, p)
        print('\n')

    def forward(self, x, y):
        
        x = self.comm_m1(x) + 5
        y = self.comm_m2(y)
        out_x = self.out_1(x)
        out_y = self.out_2(y)
        out = (out_x + out_y) / 2
        return out 


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    
    """ Init """
    x, y = torch.randn(1, 2), torch.randn(1, 2)
    label = torch.randn(1, 1)
    model = TestNet()
    creation = nn.MSELoss()
    pcgrad = PCGrad(model)
    
    
    """ Inference """
    pred = model(x, y)
    loss = creation(pred, label)
    
    loss.backward()


    print('Grad before gradient surgery')
    for name, p in model.named_parameters():
        print(name, p.grad)
        
    pcgrad.pc_grad()
    
    print('\n')
    print('Grad after gradient surgery')
    for name, p in model.named_parameters():
        print(name, p.grad)


    print('\n')
    print(pcgrad.group_grad_dict)
# class MultiHeadTestNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._linear = nn.Linear(3, 2)
#         self._head1 = nn.Linear(2, 4)
#         self._head2 = nn.Linear(2, 4)

#     def forward(self, x):
#         feat = self._linear(x)
#         return self._head1(feat), self._head2(feat)


# if __name__ == '__main__':

#     # fully shared network test
#     torch.manual_seed(4)
#     x, y = torch.randn(2, 3), torch.randn(2, 4)
#     net = TestNet()
#     y_pred = net(x)
#     pc_adam = PCGrad(optim.Adam(net.parameters()))
#     pc_adam.zero_grad()
#     loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
#     loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

#     pc_adam.pc_backward([loss1, loss2])
#     for p in net.parameters():
#         print(p.grad)

#     seperated shared network test

#     torch.manual_seed(4)
#     x, y = torch.randn(2, 3), torch.randn(2, 4)
#     net = MultiHeadTestNet()
#     y_pred_1, y_pred_2 = net(x)
#     pc_adam = PCGrad(optim.Adam(net.parameters()))
#     pc_adam.zero_grad()
#     loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
#     loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

#     pc_adam.pc_backward([loss1, loss2])
