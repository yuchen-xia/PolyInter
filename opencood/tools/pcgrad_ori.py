import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGradOri():
    def __init__(self, optimizer, model):
        self.model = model
        self._optim = optimizer

        self.gname_list = ['negotiator']
        self.group_para = {}

        for gname in self.gname_list:
            self.group_para[gname] = []

        for name, p in model.named_parameters():
            for gname in self.gname_list:
                if gname in name:
                    self.group_para[gname].append(p)

        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        input:
        - objectives: a list of objectives
        '''

        group_grads = self._pack_grad(objectives)

        for gname in self.gname_list: # 分别对每组参数进行梯度手术
            if len(group_grads[gname]) == 0: continue
            grads = group_grads[gname]
            pc_grad = self._project_conflicting(grads)
            pc_grad = self._unflatten_grad(pc_grad, gname)
            self._set_grad(pc_grad, gname)
        return


    def _pack_grad(self, objectives):
        group_grads = {}

        for gname in  self.gname_list:
            group_grads.update({gname:[]})
        
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)

            """参数组分别保存梯度"""
            for gname in self.gname_list:
                grad = self._retrieve_grad(gname)
                if len(grad) == 0: continue
                group_grads[gname].append(self._flatten_grad(grad))

        return group_grads
    
    def _retrieve_grad(self, gname):
        '''
        get the gradient of the parameters of the network.
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        '''
        grad = []
        for p in self.group_para[gname]:
            if p.grad is None: 
                continue
            grad.append(p.grad.clone())

        return grad
    

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
    

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
        for p in self.group_para[gname]:
            shape = p.grad.shape
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def _set_grad(self, grads, gname):
        idx = 0
        for p in self.group_para[gname]:
            p.grad = grads[idx]
            idx += 1
        return




class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 5)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    # torch.manual_seed(4)
    # x, y = torch.randn(2, 3), torch.randn(2, 4)
    # net = TestNet()
    # y_pred = net(x)
    # optimizer = optim.Adam(net.parameters())
    # pc_adam = PCGradOri(optimizer)
    # optimizer.zero_grad()
    # loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    # loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    # pc_adam.pc_backward([loss1, loss2])
    # for p in net.parameters():
    #     print(p.grad)

    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    y2 = torch.randn(2, 5)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGradOri(optim.Adam(net.parameters()), net)
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y2)

    pc_adam.pc_backward([loss1, loss2])
