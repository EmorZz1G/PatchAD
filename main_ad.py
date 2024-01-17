import argparse
import sys

import torch.optim

sys.path.append("..")
from data.data_loader import data_name2nc
import os
import sys

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


args = argparse.ArgumentParser()
args.add_argument("--data_name", required=False, default='MSL', type=str)
args.add_argument("--data_path", required=False, default='/home/zzj/time_series_data2/DCdetector_dataset/', type=str)

args.add_argument("--device", required=False, default='cuda:0', type=str)
args.add_argument("--ver", required=False, default='', type=str)
args.add_argument("--win_size", "-ws", required=False, default=90, type=int)
args.add_argument("--stride", "-st", required=False, default=1, type=int)
args.add_argument("--batch_size", "-bs", required=False, default=256, type=int)
args.add_argument("--epochs", '-ep', required=False, default=5, type=int)

args.add_argument('--anormly_ratio','-ar', type=float, default=1)
args.add_argument('--learning_rate','-lr', type=float, default=0.0001)
args.add_argument("--patch_sizes",'-ps', required=False, default=[3,5], type=eval)
args.add_argument("--d_model", required=False, default=40, type=int)
args.add_argument("--e_layer", required=False, default=3, type=int)

args.add_argument("--seed", required=False, default=0, type=int)
args.add_argument("--save_model", required=False, default=1, type=int)
args.add_argument("--full_res", required=False, default=1, type=int)
args.add_argument("--model_save_path", required=False, default=f"/home/zzj/patch_ad_model2/", type=str)
args.add_argument(
    "--res_pth", required=False, default=f"/home/zzj/patch_ad_result2/", type=str
)
args.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
args.add_argument("--patch_mx", required=False, default=0.3, type=float)



args.add_argument('--index', type=int, default=137)
args.add_argument('--input_c', type=int, default=38)

params = args.parse_args().__dict__

# post preceessing
nc = data_name2nc(params['data_name'])
params['input_c'] = nc
patch_sizes = [int(patch_index) for patch_index in params['patch_sizes']]
patch_sizes = sorted(patch_sizes)
params['patch_sizes'] = patch_sizes
if params['ver'] != '':
    params['model_save_path'] = os.path.join(params['model_save_path'],params['ver'])
    params['res_pth'] = os.path.join(params['res_pth'],params['ver'])


def print_params(params_):
    s = ""
    for k,v in params_.items():
        if k in ['pth','res_pth','save_model','test_mode','epochs','ar']:continue
        s += str(k)+":"+str(v)+"_"
    print('file_name',s)
    return s



from logging import getLogger, basicConfig
import logging
import os
import pickle
from utils2.utils2 import seed_all
from trainer.patchad_trainer import Solver as Trainer # 
from metrics.metrics import my_combine_all_evaluation_scores
from torch.backends import cudnn
cudnn.benchmark = True

def main(params):
    cudnn.benchmark = True
    os.makedirs(params['model_save_path'], exist_ok=True)
    solver = Trainer(params)

    if params['mode'] == 'train':
        print('========Train========')
        solver.train()
        print('========Test========')
        solver.test(1)
        # metrics_calculate(score,y,step=200)
    elif params['mode'] == 'test':
        print('========Test========')
        solver.test(1)
    elif params['mode'] == 'plot':
        print('========Plot========')
        solver.test(1)

    return solver

from datetime import datetime
if __name__ == "__main__":
    res_pth = os.path.join(params['res_pth'], params['data_name'])
    os.makedirs(res_pth, exist_ok=True)
    if params['ver'] != '':
        fi_name = '/'+params['ver']+'_result_v5.log'
        add_flag = True

    elif params['data_name'] in ['UCR','UCR_AUG']:
        fi_name = '/'+'result2.log'
        add_flag = True
    else:
        fi_name = '/'+datetime.now().strftime("%Y-%m-%d-%H_%M_%S")+'.log'
        add_flag = True
    sys.stdout = Logger(res_pth+fi_name, add_flag, sys.stdout)
    print("Logger has been set up.")
    print("Logger has been set up.")
    print("Logger has been set up.")
    for key, value in params.items():
        print(f'{key}:{value}')

    device = params["device"]
    seed = params["seed"]
    
    ar = params['anormly_ratio']
    if seed > 0 :seed_all(seed)
    epochs = params["epochs"]
    save_model = params["save_model"]
    ver = params["ver"]
    pth = params["model_save_path"] + ver
    print("Result path", res_pth)
    
    main(params)
    



    
