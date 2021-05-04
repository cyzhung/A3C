# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:48:22 2021

@author: 123
"""
from env import create_env
import torch
from models import ActorCritic 
import torch.nn.functional as F
import time
def push_and_pull(opt, local_model, global_model, loss):
    #p=([x.grad for x in opt.param_groups[0]['params']])
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    
    for lp, gp in zip(local_model.parameters(), global_model.parameters()):
        gp._grad = lp.grad
    opt.step()
    #p2=([x.grad for x in opt.param_groups[0]['params']])
    
    #print(p==p2)


        
    #print([x.grad for x in opt.param_groups[0]['params']])
    # pull global parameters
    local_model.load_state_dict(global_model.state_dict())
    
def test(global_model,device,args,delay=0):
    world=args.world
    stage=args.stage
    env = create_env(world,stage)
    device=device
    state=env.reset()
    state=(env.reset()).to(device,dtype=torch.float)

    state=state.view(1,1,80,80)
    done=True
    model=ActorCritic().to(device)
    model.eval()
    model.load_state_dict(global_model.state_dict())
    
    while(True):
        if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
            
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        env.render()
        p,_,h_0,c_0=model(state,h_0,c_0)
        policy=F.softmax(p,dim=1)
        action=torch.argmax(policy)
        
        #print(policy)
        #action=m.sample()

        
        next_state, _, done, info = env.step(action.item())
        
        next_state=(next_state).to(device,dtype=torch.float)
        next_state=next_state.view(1,1,80,80)


            
            
        #self.log_probs.append(log_prob)
        #self.values.append(value)
       # self.rewards.append(reward)
        

        

        state=next_state
        if(done):
            if info['flag_get']:
                break
            state=env.reset()
            state=state.to(device)
            state=state.view(1,1,80,80)
            model.load_state_dict(global_model.state_dict())
        time.sleep(delay)
    print('Success clear {}-{}'.format(world,stage))