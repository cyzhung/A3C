# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:22:03 2021

@author: 123
"""

from models import ActorCritic
import torch.multiprocessing as mp
import torch.optim as optim
from worker import Worker
from utility import local_test
from ShareAdam import SharedAdam
import torch
import os
import argparse
torch.set_num_threads(1)
NUM_PROCESSES=10
device="cpu"

                        

                
         
                



               

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--load_model",type=bool,default=False)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=get_args()
    LEVEL=str(args.world)+'-'+str(args.stage)

    folder='./model/{}'.format(LEVEL)

    if(not  os.path.exists(folder)):
        os.mkdir(folder)
    
    global_model=ActorCritic()
    optimizer=SharedAdam(global_model.parameters(),lr=1e-4)

    PATH='./model/{}/A3C_{}.pkl'.format(LEVEL,LEVEL) 
    epoch=1
    if(args.load_model):
        if(os.path.exists(PATH)):
            print('Loaded Model')
            check_point=torch.load(PATH)
            global_model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            epoch=check_point['epoch']
    
    global_model.share_memory()
    
    workers=[Worker(0,global_model,optimizer,device,args,epoch,True)]
    test_worker=[mp.Process(target=local_test,args=(global_model,device,args))]
    
    workers=workers+[Worker(i,global_model,optimizer,device,args,epoch) for i in range(1,NUM_PROCESSES)]
    workers=workers+test_worker
    [w.start() for w in workers]
    
    

    #workers.append(test_worker)
    
    [w.join() for w in workers]
    
    
    

    
    
    