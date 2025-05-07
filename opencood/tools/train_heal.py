# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import threading
import time

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=0)
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--run_test', type=str, default='True',
                        help='whether run inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(train_loader)
    
    modality_type_list = set(hypes['heter']['mapping_dict'].values())

    origin_modality_model_dir = {}
    for modality_name in modality_type_list:
        origin_modality_model_dir[modality_name] = os.path.join(opt.model_dir, \
                                                                hypes['model']['args'][modality_name]['model_dir'])

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model, opt.eval_epoch)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")
        
        if init_epoch == 0:
            for modality_name in modality_type_list:
                _, model = train_utils.load_modality_saved_model(\
                    origin_modality_model_dir[modality_name], \
                    model, modality_name)   
            _, model = train_utils.load_saved_model_sub(saved_path, model, 'shared')         

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        # model.train()
        # try: # heter_model stage2
        #     model.model_train_init()
        # except:
        #     print("No model_train_init function")
            
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            # reset loss computation logic
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)

            if supervise_single_flag:
                # 对Forground Estimator 进行监督
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", pbar=pbar2)

            # back-propagation
            final_loss.backward()
            optimizer.step()
            
            pbar2.update(1)

            # torch.cuda.empty_cache()  # it will destroy memory buffer
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
            
            # break

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            train_utils.save_sub_model(saved_path, 'shared', model, epoch)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())
                    
                    # break

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                    
                train_utils.save_sub_model(saved_path, 'shared', model, epoch, besval='True', lowest_val_epoch=lowest_val_epoch)
                    
                for modality_name in modality_type_list:                    
                    train_utils.save_model_back_to_local(\
                        origin_modality_model_dir[modality_name], model, modality_name)
                    # torch.save(modality_checkpoint_name[modality_name], modality_model_include_comm)
                lowest_val_epoch = epoch + 1


            run_test = True
            if (opt.run_test == 'True') or (opt.run_test == 'true') or (opt.run_test == '1'):
                run_test = True
            elif (opt.run_test == 'False') or (opt.run_test == 'false') or (opt.run_test == '0'):
                run_test = False
                
            if run_test:
                epoch_th = threading.Thread(target=epoch_test, args=(opt, hypes, epoch))
                epoch_th.start()



        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)




def epoch_test(opt, hypes, epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    modes = list(set(hypes['heter']['mapping_dict'].values()))
    modes.insert(0, 'hete')
    for mode in modes:
        cmd = f"python opencood/tools/inference.py \
            --model_dir {opt.model_dir} \
            --eval_epoch {epoch + 1} \
            --collab_mode {mode}"
        print(f"Running command: {cmd}")
        os.system(cmd)
        time.sleep(0.1)

    with open(os.path.join(opt.model_dir, 'result.txt'), 'a+') as f:
        f.write('\n')
    time.sleep(0.1)

if __name__ == '__main__':
    main()
