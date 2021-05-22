# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:22:03 2021

@author: 123
"""

from models import ActorCritic
from models import ActorCritic_LSTM
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
    parser.add_argument("--world", type=str, default=1)
    parser.add_argument("--stage", type=str, default=1)

    args = parser.parse_args()
    return args

modelType_dict={'1-1':'LSTM',
'1-2':'normal',
'1-4':'LSTM',
'2-1':'normal',
'2-2':'LSTM',
'3-1':'LSTM',
'7-2':'LSTM',
'8-3':'LSTM'
}



    

if __name__=='__main__':    
    args=get_args()
    LEVEL=args.world+'-'+args.stage
    model_type=modelType_dict[LEVEL]
    if(model_type == "LSTM"):
        global_model=ActorCritic_LSTM()
    else:
        global_model=ActorCritic()

    optimizer=SharedAdam(global_model.parameters(),lr=1e-4)
    
    world=args.world
    stage=args.stage
    LEVEL=str(world)+'-'+str(stage)
    PATH='./model/{}/A3C_{}_{}.pkl'.format(LEVEL,LEVEL,model_type)
    if(os.path.exists(PATH)):
        print('Loaded Model')
        check_point=torch.load(PATH)

        

        
        global_model.load_state_dict(check_point['model_state_dict'])
        
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        global_model.share_memory()
    
        global_test(global_model,device,args,model_type)
        
    else:
        print("model {} does not exist!".format(LEVEL))

    

    
    

    
    
    