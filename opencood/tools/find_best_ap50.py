# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import threading
import time

import torch

from opencood.tools.pcgrad import PCGrad
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

import wandb
os.environ["WANDB_API_KEY"] = 'b058297d8947bc34e6e11764cffa6a3a94671dc6'
os.environ["WANDB_MODE"] = "offline"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        help='data generation yaml file needed ')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=-1)
    parser.add_argument("--stage", "-s", type=str,
                        help='Traning stage. \
                        nego: negotiate common feature space. \
                        ft: Downstream task adaptation. \
                        nego: nsd and nsd.')
    parser.add_argument('--lsd_mode', '-lsd', type=int, default=0,
                        help='whether use lsd config')
    parser.add_argument('--model_dir', default='', required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--run_test', type=str, default='True',
                        help='whether run inference.')
    parser.add_argument('--use_cb', type=str, default='False',
                        help='whether collaborate in public or local sematic space')
    opt = parser.parse_args()
    return opt


def main():
    """
    寻找从result.txt中寻找最大ap对应的epoch, 并将comm模块的参数保存回stage0_mx_collab
    """
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    """opt.stage: nego, add, ft"""
    if opt.stage == 'nego' or opt.stage == 'add' or opt.stage == 'unify':
        hypes['model']['args']['comm_space'] = 'pub'
        hypes['model']['core_method'] = hypes['model']['core_method']['train']
        hypes['loss']['core_method'] = hypes['loss']['core_method']['nego']
        hypes['model']['args'].update({'stage': opt.stage})
        
    train_setting = ['train_params', 'optimizer', 'lr_scheduler']
    for st in train_setting:
        hypes[st] = hypes['train_setting'][opt.stage][st]
    
    # print('Dataset Building')
    # opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    # opencood_validate_dataset = build_dataset(hypes,
    #                                           visualize=False,
    #                                           train=False)

    # train_loader = DataLoader(opencood_train_dataset,
    #                           batch_size=hypes['train_params']['batch_size'],
    #                           num_workers=4,
    #                           collate_fn=opencood_train_dataset.collate_batch_train,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           drop_last=True,
    #                           prefetch_factor=2)
    # val_loader = DataLoader(opencood_validate_dataset,
    #                         batch_size=hypes['train_params']['batch_size'],
    #                         num_workers=4,
    #                         collate_fn=opencood_train_dataset.collate_batch_train,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         drop_last=True,
    #                         prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 除comm模块均为推理状态
    # model.eval()
    # for name, sub_model in model.named_children():
        # if 'comm' in name:
            # sub_model.train()
    # model.backbone_m1.training
    # record lowest validation loss checkpoint.
    # lowest_val_loss = 1e5
    # lowest_val_epoch = -1
    
    #   lowest_val_loss = 1e5

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    # num_steps = len(val_loader)

    modality_type_list = set(hypes['heter']['mapping_dict'].values())

    # if we want to train from last checkpoint.
    
    if opt.model_dir:
        saved_path = opt.model_dir
        
        """加载预训练模型参数, 若已加入联盟, 加载comm模块参数"""
        for modality_name in modality_type_list:
            # Add local path to pub_codebook path and local modality model path, and get full path
            modality_hypes = hypes['model']['args'][modality_name]
            modality_hypes['model_dir'] = os.path.join(opt.model_dir, modality_hypes['model_dir'])
            load_comm = hypes['model']['args'][modality_name]['allied']
            _, model = train_utils.load_modality_saved_model_ap(\
                hypes['model']['args'][modality_name]['model_dir'], \
                model, modality_name, load_comm)
        
        """若文件夹下有协商训练几轮后的模型, 使用训练好的模型参数覆盖comm部分"""
        if opt.eval_epoch == -1:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model)
        else:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model, opt.eval_epoch)
                
        # modality_checkpoint_name = {}
        # if init_epoch == 0:
        #     # 若文件夹下无预训练模型, 从每个模态的文件夹加载预训练参数, 从pub_codebook加载公共码本            
        #     for modality_name in modality_type_list:
        #         load_comm = hypes['model']['args'][modality_name]['allied']
        #         _, model = train_utils.load_modality_saved_model_ap(\
        #             hypes['model']['args'][modality_name]['model_dir'], \
        #             model, modality_name, load_comm)
            
                
        # elif init_epoch == 1: # 参与协商智能体中, 有已经allied的时(即stage1联盟内协商已完成), 训练从epoch1开始
        #     for modality_name in modality_type_list:
        #         # 若未在联盟中, 加载该模态编码器预训练参数
        #         if not hypes['model']['args'][modality_name]['allied']: 
        #             _, model = train_utils.load_modality_saved_model_ap(\
        #             hypes['model']['args'][modality_name]['model_dir'], \
        #             model, modality_name, False)

        lowest_val_epoch = init_epoch
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps, init_epoch=init_epoch)
        # print(f"resume from {init_epoch} epoch.")

    else:
        assert opt.model_dir

    
    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)
    # model.
    # record training
    # In train, record every step in every epoch
    # In validation, record mean loss of all steps in every epoch 
    # writer = SummaryWriter(saved_path)    
    
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=hypes["name"]+time.strftime('%m_%d_%H_%M_%S'),
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": hypes["optimizer"]["lr"],
    #     "architecture": hypes["name"],
    #     "dataset": hypes["root_dir"],
    #     "epochs": hypes["train_params"]["epoches"],
    #     }
    # )

    print('Validate start')
     
    modes = list(set(hypes['heter']['mapping_dict'].values()))
    modes = model.newtype_modality_list


    result_file = os.path.join(opt.model_dir, 'result.txt')
    if not os.path.exists(result_file):
        f = open(result_file, 'w')
        f.close()
    max_ap50, max_ap50_epoch = 0, -1
    # max_ap50, max_ap50_epoch = find_best_pub_ap50(result_file, modes)

    ap50, ap50_epoch = find_best_pub_ap50(os.path.join(opt.model_dir, 'result.txt'), modes)



    if ap50 > max_ap50:
        init_epoch, model = train_utils.load_saved_model_ap(saved_path, model, opt.eval_epoch)
        torch.save(model.state_dict(),
            os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (ap50_epoch)))
        if max_ap50_epoch != -1 and os.path.exists(os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (max_ap50_epoch))):
            os.remove(os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (max_ap50_epoch)))            
        
        for modality_name in modality_type_list:                    
            train_utils.save_model_back_to_local(\
                hypes['model']['args'][modality_name]['model_dir'], model, modality_name)


def find_best_pub_ap50(file_name, mode_list):
    """
    从文件目录下的result.txt文件中提取最大AP @0.5值和对应的epoch
    :param file_name: 文件名称
    :return: 包含模型名和对应AP值的字典
    """
    f = open(file_name, 'r', encoding='utf-8')
    file_content = f.read()
    f.close()
    
    epoch_results = len(mode_list) + 1
    
    # 将文件内容按行分割
    lines = file_content.strip().split('\n')

    max_ap50, best_ap50_epoch = 0, -1
    if len(lines) < epoch_results:
        return max_ap50, best_ap50_epoch
    
    max_ap50 = 0
    best_ap50_epoch = -1
    lidx = 0
    while lidx < len(lines):
        # mode = mode_list[idx]
        msg_mode = lines[lidx]
        
        if 'ind' in msg_mode: 
            lidx = lidx + epoch_results + 1
            continue

        locate_ap50 = msg_mode.index('@0.5:')
        ap50 = float(msg_mode[locate_ap50+7 : locate_ap50+13])

        if ap50 >  max_ap50:
            max_ap50 = ap50
            locate_epoch = msg_mode.index('Epoch: ')
            best_ap50_epoch = int(msg_mode[locate_epoch+7 : locate_epoch + 9])
            # print(msg_mode[locate_epoch+7 : locate_epoch + 9])
        lidx = lidx + epoch_results + 1

    return max_ap50, best_ap50_epoch


if __name__ == '__main__':
    main()
