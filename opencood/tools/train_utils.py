# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim

def load_pretrained_model(saved_path, model):
    """
    加载model模型的参数
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    model_file = saved_path
    print('load pretrained model by {}'.format(model_file) )
    checkpoint = torch.load(
        model_file,
        map_location='cpu')
    
    load_result = model.load_state_dict(checkpoint, strict=False)
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    # 打印没有加载的参数（模型中存在，但在 state_dict 中不存在）
    # assert len(load_result.missing_keys) == 0
    # print("Missing keys:", load_result.missing_keys)

    # 打印意外的参数（state_dict 中存在，但在模型中不存在）
    # print("Unexpected keys:", load_result.unexpected_keys)

    del checkpoint

    return model

def create_encoder(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['core_method']
    backbone_config = hypes['args']

    model_filename = "opencood.models.sub_modules." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('encoder not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance

def load_pretrained_submodule(saved_path, model, name):
    """
    加载model中名为name的子模块的参数
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    model_file = saved_path
    print('load pretrained model by {}'.format(model_file) )
    checkpoint = torch.load(
        model_file,
        map_location='cpu')
    selected_param = {}
    for k, v in checkpoint.items():
        # print(k)
        if k.startswith(f"{name}."):
            idx = k.index(".")
            k = k[idx+1:]
            selected_param[k] = v
    
    load_result = model.load_state_dict(selected_param, strict=False)
    # 打印没有加载的参数（模型中存在，但在 state_dict 中不存在）
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    # print("Missing keys:", load_result.missing_keys)

    # 打印意外的参数（state_dict 中存在，但在模型中不存在）
    # print("Unexpected keys:", load_result.unexpected_keys)

    del checkpoint

    return model


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def check_missing_key(model_state_dict, ckpt_state_dict, modality = None):
    if modality is not None:
        model_keys = []
        checkpoint_keys = []
        for k in model_state_dict.keys():
            if modality in k:
                model_keys.append(k)
                checkpoint_keys.append(k)
        model_keys = set(model_keys)
        checkpoint_keys = set(checkpoint_keys)
    else:
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(ckpt_state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    missing_key_modules = set([(keyname.split('.')[0], keyname.split('.')[1]) for keyname in missing_keys])
    extra_key_modules = set([keyname.split('.')[0] for keyname in extra_keys])

    print("------ Loading Checkpoint ------")
    if len(missing_key_modules) == 0 and len(extra_key_modules) ==0:
        return

    print("Missing keys from ckpt:")
    print(*missing_key_modules,sep='\n',end='\n\n')
    # print(*missing_keys,sep='\n',end='\n\n')

    print("Extra keys from ckpt:")
    print(*extra_key_modules,sep='\n',end='\n\n')
    print(*extra_keys,sep='\n',end='\n\n')

    print("You can go to tools/train_utils.py to print the full missing key name!")
    print("--------------------------------")


def load_saved_model(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    if (epoch is None) or (epoch == 0):
        file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("resuming best validation model at epoch %d" % \
                    eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            check_missing_key(model.state_dict(), loaded_state_dict)
            print('If missing keys about comm_mx, ignored. Check whether comm_mx is missing from the next message!')
            model.load_state_dict(loaded_state_dict, strict=False)
            return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = epoch 
        
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                        'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model

def load_saved_model_ap(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestap_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("resuming best validation model at epoch %d" % \
                    eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestap_at")))
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            check_missing_key(model.state_dict(), loaded_state_dict)
            print('If missing keys about comm_mx, ignored. Check whether comm_mx is missing from the next message!')
            model.load_state_dict(loaded_state_dict, strict=False)
            return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestap_at")), model
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = epoch 
        
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                        'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model

def save_model_back_to_local(saved_path, model, modality_name):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise FileNotFoundError('checkpoints not found!')
            
        return f'net_epoch{initial_epoch_}.pth'


    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        #  = eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at"))   
        file_name = file_list[0].split("/")[-1]
    else:
        file_name = findLastCheckpoint(saved_path)
    
    
    modality_model_include_comm = {}
    for key, para in model.state_dict().items():
        if modality_name in key:
            modality_model_include_comm[key] = para
    
    
    torch.save(modality_model_include_comm, os.path.join(saved_path, file_name))
    
    
def save_model_back_to_local_ap(saved_path, model, modality_name):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise FileNotFoundError('checkpoints not found!')
            
        return f'net_epoch{initial_epoch_}.pth'


    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestap_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        #  = eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at"))   
        file_name = file_list[0].split("/")[-1]
    else:
        file_name = findLastCheckpoint(saved_path)
    
    
    modality_model_include_comm = {}
    for key, para in model.state_dict().items():
        if modality_name in key:
            modality_model_include_comm[key] = para
    
    
    torch.save(modality_model_include_comm, os.path.join(saved_path, file_name))
    


def load_modality_saved_model_ap(saved_path, model, modality_name, allied=False, epoch=None):
    """
    Load saved modality model parameters if exiseted 
    Allied in hypes for loaed comm parameters or not

    Parameters
    __________
    hypes : str
       modality setting parameters
       
    modality_name : str
        name of modality
        
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    # saved_path = hypes['model_dir']
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    file_name = None

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    def findModalityParameters(loaded_state_dict):
        loaded_keys = []
        load_comm = allied  # 若在联盟中, 则加载comm模块参数
        for key, para in loaded_state_dict.items():
            if modality_name in key: 
                if 'comm' in key and (not load_comm): # 若不加载comm参数, 跳过
                    continue
                loaded_keys.append(key)
        del_keys = list(set(loaded_state_dict.keys()) - set(loaded_keys))
        for key in del_keys:
            loaded_state_dict.pop(key)
            
        return loaded_state_dict

    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestap_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            best_epoch = eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestap_at"))
            print("resuming best validation model at epoch %d" % \
                     best_epoch)
            file_name = file_list[0]
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            loaded_state_dict = findModalityParameters(loaded_state_dict)   
            check_missing_key(model.state_dict(), loaded_state_dict, modality_name)
            print('If missing keys about comm_mx, ignored. Check whether comm_mx is missing from the next message!')
            model.load_state_dict(loaded_state_dict, strict=False)
            return  best_epoch, model
        
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = epoch 
    
    file_name = os.path.join(saved_path, f'net_epoch{initial_epoch}')
        
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                        'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        

        loaded_state_dict = findModalityParameters(loaded_state_dict)   

             
        check_missing_key(model.state_dict(), loaded_state_dict, modality_name)
        model.load_state_dict(loaded_state_dict, strict=False)

        if allied:
            print('Load Comm')

    return initial_epoch, model  

def load_modality_saved_model(saved_path, model, modality_name, allied=False, epoch=None):
    """
    Load saved modality model parameters if exiseted 
    Allied in hypes for loaed comm parameters or not

    Parameters
    __________
    hypes : str
       modality setting parameters
       
    modality_name : str
        name of modality
        
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    # saved_path = hypes['model_dir']
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    file_name = None

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    def findModalityParameters(loaded_state_dict):
        loaded_keys = []
        load_comm = allied  # 若在联盟中, 则加载comm模块参数
        for key, para in loaded_state_dict.items():
            if modality_name in key: 
                if 'comm' in key and (not load_comm): # 若不加载comm参数, 跳过
                    continue
                loaded_keys.append(key)
        del_keys = list(set(loaded_state_dict.keys()) - set(loaded_keys))
        for key in del_keys:
            loaded_state_dict.pop(key)
            
        return loaded_state_dict

    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            best_epoch = eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at"))
            print("resuming best validation model at epoch %d" % \
                     best_epoch)
            file_name = file_list[0]
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            loaded_state_dict = findModalityParameters(loaded_state_dict)   
            check_missing_key(model.state_dict(), loaded_state_dict, modality_name)
            print('If missing keys about comm_mx, ignored. Check whether comm_mx is missing from the next message!')
            model.load_state_dict(loaded_state_dict, strict=False)
            return  best_epoch, model
        
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = epoch 
    
    file_name = os.path.join(saved_path, f'net_epoch{initial_epoch}')
        
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                        'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        

        loaded_state_dict = findModalityParameters(loaded_state_dict)   

             
        check_missing_key(model.state_dict(), loaded_state_dict, modality_name)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model

def load_saved_model_comm(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    
    saved_path = os.path.join(saved_path, 'comm_module')
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    comm_dict = dict()
    for k in model.state_dict().keys():
        if ('focuser' in k) or ('comm' in k):
            comm_dict[k] = model.state_dict()[k]
    
    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, 'comm_epoch_bestval_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("resuming best comm validation model at epoch %d" % \
                    eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("comm_epoch_bestval_at")))
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            check_missing_key(comm_dict, loaded_state_dict)
            model.load_state_dict(loaded_state_dict, strict=False)
            return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("comm_epoch_bestval_at")), model
        
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = int(epoch)

    # initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming comm by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'comm_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(comm_dict, loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def save_sub_model(save_path, sub_name, model, epoch, besval = None, lowest_val_epoch = None):
    """
    Save path: checkpoint dir
    sub_name: sub module name
    
    Save to /checkpoint/{sub_name}_net
    """
    
    sub_model_dict = {}
    for k, v in model.state_dict().items():
        if sub_name in k:
            sub_model_dict.update({k: v})

    sub_path = os.path.join(save_path, f'{sub_name}_net')
    
    
    if besval is None:
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
            
        save_file = os.path.join(sub_path, f'{sub_name}net_epoch{epoch + 1}.pth')

        torch.save(sub_model_dict, save_file)
    
    else:
        torch.save(model.state_dict(),
        os.path.join(sub_path,
                    f'{sub_name}net_epoch_bestval_at{(epoch + 1)}.pth'))
        if lowest_val_epoch != -1 and os.path.exists(os.path.join(sub_path,
                            f'{sub_name}net_epoch_bestval_at{lowest_val_epoch}.pth')):
            os.remove(os.path.join(sub_path,
                            f'{sub_name}net_epoch_bestval_at{lowest_val_epoch}.pth'))    


def load_saved_model_sub(saved_path, model, sub_name, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    
    saved_path = os.path.join(saved_path, f'{sub_name}_net')
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    sharednet_dict = dict()
    for k in model.state_dict().keys():
        if sub_name in k:
            sharednet_dict[k] = model.state_dict()[k]
    
    if epoch is None:
        file_list = glob.glob(os.path.join(saved_path, f'{sub_name}net_epoch_bestval_at*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("resuming best shared validation model at epoch %d" % \
                    eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip(f"{sub_name}net_epoch_bestval_at")))
            loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
            check_missing_key(sharednet_dict, loaded_state_dict)
            model.load_state_dict(loaded_state_dict, strict=False)
            return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip(f"{sub_name}net_epoch_bestval_at")), model
        
        initial_epoch = findLastCheckpoint(saved_path)
    else:
        initial_epoch = int(epoch)

    # initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming shared by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         f'{sub_name}net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(sharednet_dict, loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def find_model_comm(model_dict):
    # 模型中的comm从模型字典中寻找相关参数
    model_comm_dict = {}
    for k, v in model_dict.items():
        if ('comm' in k) or ('focuser' in k):
            model_comm_dict.update({k: v})
    
    return model_comm_dict
    
def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../../checkpoints')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        
    # backup_script(full_path)
    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)
    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for epoch in range(last_epoch):
        if lr_schedule_config['core_method'] != 'cosineannealwarm':
            scheduler.step()
        if lr_schedule_config['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * n_iter_per_epoch + 0)

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)

def load_pub_cb(model, epoch):
    """
    Change filename [xxx/pub_codebook.pth] to [xxx/pub_cb_epoch{N}.pth]
    and load file
    """
    pub_cb_name = model.pub_cb_path
    pub_cb_name = pub_cb_name.replace('.pth', f'_epoch{epoch}.pth')
    pub_cb_name = pub_cb_name.replace('codebook_', 'cb_')
    model.init_pub_codebook = torch.load(pub_cb_name)
    model.init_pub_codebook.requires_grad = False
    
    return model

def save_pub_cb(model, epoch):
    pub_cb_name = model.pub_cb_path
    pub_cb_name = pub_cb_name.replace('.pth', f'_epoch{epoch}.pth')
    pub_cb_name = pub_cb_name.replace('codebook_', 'cb_')
    torch.save(model.pub_codebook, pub_cb_name)

def save_pub_query_emb(model, epoch):
    pub_query_name = model.pub_query_emb_path
    pub_query_name = pub_query_name.replace('.pth', f'_epoch{epoch}.pth')
    # pub_cb_name = pub_cb_name.replace('codebook_', 'cb_')
    torch.save(model.pub_query_embeddings, pub_query_name)