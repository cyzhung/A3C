# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:22:03 2021

@author: 123
"""

from models import ActorCritic
import torch.multiprocessing as mp
import torch.optim as optim
from worker import Worker
from utility import global_test
from ShareAdam import SharedAdam
import torch
import os
import argparse
torch.set_num_threads(1)
device="cpu"
           
                        
                
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    return args


    

if __name__=='__main__':    
    args=get_args()
    global_model=ActorCritic()
    optimizer=SharedAdam(global_model.parameters(),lr=1e-4)
    
    world=args.world
    stage=args.stage
    LEVEL=str(world)+'-'+str(stage)
    PATH='./model/{}/A3C_{}.pkl'.format(LEVEL,LEVEL)
    if(os.path.exists(PATH)):
        print('Loaded Model')
        check_point=torch.load(PATH)
        global_model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        global_model.share_memory()
    
        global_test(global_model,device,args)

    else:
        print("model {} does not exist!".format(LEVEL))

    

    
    

    
    
    