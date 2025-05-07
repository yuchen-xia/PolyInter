from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGradFast():
    def __init__(self, model, hypes):
        # self.model = model
        self.nego_modality_list = model.newtype_modality_list
        self.group = hypes['train_setting']['pc_grad_group']
        
        """参数分组"""
        self.modality_negotiator = {}

        modality_flag = self.nego_modality_list[0]
        if self.group:
            # layer and estimators
            for layer_idx in range(0, eval(f'model.negotiator_bak_{modality_flag}.resnet.layernum')):
                self.gname_list.append(f'layer{layer_idx}')    
            
            # deblock and output
            self.gname_list.extend(['shrink_header', 'deblocks']) 
        else:
            self.gname_list = ['e']

        """分别向group分组中添加对应的参数, 不同模态的参数排列为paras"""
        self.group_paras = OrderedDict() # {gname: [[para_mode1], [para_mode2]]}
        for gname in self.gname_list:
            self.group_paras[gname] = []
            for mode in self.nego_modality_list:
                self.group_paras[gname].append([])


        for name, para in model.named_parameters():
            if ('negotiator_bak' in name):
                for gname in self.gname_list:
                    if (gname in name):
                        for idx in range(len(self.nego_modality_list)):
                            mode = self.nego_modality_list[idx]
                            if mode in name:
                                self.group_paras[gname][idx].append(para)

            
    def grad_surgery(self):
        '''
        input:
        - objectives: a list of objectives
        '''

        group_grads = self._pack_grad()
        for gname in self.gname_list:
            grads = group_grads[gname]
            if len(grads) == 0: continue
            pc_grad = self._project_conflicting(grads)
            pc_grad = self._unflatten_grad(pc_grad, gname)
            self._set_grad(pc_grad, gname)

        return
        
    def _retrieve_grad(self, gname):
        '''
        get the gradient of the parameters of the network.
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        '''
        grads = []
        for p_list in self.group_paras[gname]:
            grad = []
            for p in p_list:
                if p.grad is None: 
                    continue
                grad.append(p.grad.clone().flatten())
            grad = torch.cat(grad)
            grads.append(grad)

        return grads

    def _pack_grad(self):
        group_grads = {}
        for gname in self.gname_list:
            group_grads.update({gname: self._retrieve_grad(gname)})

        return group_grads
                    


    def _project_conflicting(self, grads):
        """Core Function. 遍历每个任务产生的梯度, 消除梯度冲突, 返回所有任务的平均值"""
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        pc_grad = torch.stack(pc_grad).mean(dim=0)
        return pc_grad


    def _unflatten_grad(self, grads, gname):
        """根据参数找到梯度形状, 将梯度还原回原来的大小"""
        unflatten_grad, idx = [], 0
        for p in self.group_paras[gname][0]:
            shape = p.grad.shape
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def _set_grad(self, grads, gname):

        for g_idx in range(len(grads)):
            for idx in range(len(self.nego_modality_list)):
                self.group_paras[gname][idx][g_idx].grad = grads[g_idx]

        return
    


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


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


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
