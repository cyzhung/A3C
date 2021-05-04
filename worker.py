# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:36:31 2021

@author: 123
"""
import torch
from models import ActorCritic 
from env import create_env

import torch.nn.functional as F

import os
import time
from torch.distributions import Categorical
import random
from utility import push_and_pull
import torch.multiprocessing as mp







def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
TAU=1.0




class Worker(mp.Process):
    def __init__(self,i,global_model,opt,device,args,save=False):
        super(Worker,self).__init__()
        self.name='Worker%i'%i
        self.device=device
        self.AC=None


        self.log_probs=[]
        self.values=[]
        self.rewards=[]
        #self.states=[]
        self.entropies=[]
        self.GAMMA=0.9
        self.optimizer=opt
        self.global_model=global_model
        self.save=save
        self.stage=args.stage
        self.world=args.world
        self.level=str(self.world)+'-'+str(self.stage)
    def run(self):
        #self.global_model=self.global_model.to(self.device)
        torch.manual_seed(random.randint(1,1000))
        self.AC=ActorCritic()
        #optimizer_to(self.optimizer,self.device)
        env = create_env(self.world,self.stage)
        state=(env.reset())
        #state=state.reshape(1,1,80,80)
        state=(state).to(self.device,dtype=torch.float)

        #state=self.imageProcess(state) 
        Timestamp=50
        i_epoch=1

        done=True
        while True:
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
                
            h_0 = h_0.to(self.device)
            c_0 = c_0.to(self.device)

            t=0
            for i in range((Timestamp)):
                env.render()
                    
                p,value,h_0,c_0=self.AC(state,h_0,c_0)


                
                policy=F.softmax(p,dim=1)
                log_prob=F.log_softmax(p,dim=1)
                entropy=-(policy*log_prob).sum(1,keepdim=True)
                
                m=Categorical(policy)

                action=m.sample()

                next_state, reward, done, info = env.step(action.item())

                #reward=reward/15
                

                #next_state=next_state.view(1,1,80,80)
                next_state=(next_state).to(self.device,dtype=torch.float)
                

                
                #self.states.append(state)
                self.log_probs.append(log_prob[0,action])
                self.rewards.append(reward)
                self.values.append(value)
                self.entropies.append(entropy)
                
                state=next_state
                
                t+=1
                
                if(done):
                    state=(env.reset())
                    #state=state.reshape(1,1,80,80)
                    state=state.to(self.device)
                    #state=self.imageProcess(state)
                    break
            """
            actor_loss=0
            critic_loss=0
            returns=[]
            R=0
            for reward in self.rewards[::-1]:
                R=reward+self.GAMMA*R
                returns.insert(0,R)
            """
            #td=torch.tensor([1],dtype=torch.float).to(device)
            
            R = torch.zeros((1, 1), dtype=torch.float)
            if not done:
                _, R, _, _ = self.AC(state, h_0, c_0)

            R=R.to(self.device)
            actor_loss=0
            critic_loss=0
            entropy_loss=0
            advantage=torch.zeros((1, 1), dtype=torch.float)
            advantage=advantage.to(self.device)
            next_value=R
                
            for log_prob,reward,value,entropy in list(zip(self.log_probs,self.rewards,self.values,self.entropies))[::-1]:
                advantage=advantage*self.GAMMA
                advantage=advantage+reward+self.GAMMA*next_value.detach()-value.detach()
                next_value=value
                actor_loss=actor_loss+(-log_prob*advantage)
                R=R*self.GAMMA+reward
                critic_loss=critic_loss+(R-value)**2/2
                entropy_loss=entropy_loss+entropy


            
            total_loss=actor_loss+critic_loss-0.01*entropy_loss
            
            
            push_and_pull(self.optimizer, self.AC, self.global_model, total_loss)
            #print([x.grad for x in self.optimizer.param_groups[0]['params']])
           # for name, parms in self.C.named_parameters():	
              #  print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)

            
            if(i_epoch%10==0):
                print(self.name+"\ Episode %d \ Actor loss:%f \ Critic Loss:%f \ Total Loss: %f"%(i_epoch,actor_loss.item(),critic_loss.item(),total_loss.item()))
            
            

            """
            y.append(critic_loss.item())
            x.append(i_epoch)
            plt.plot(x,y) #畫線
            plt.show() #顯示繪製的圖形
            """                    
            i_epoch+=1
            
            del self.log_probs[:]
            del self.rewards[:]
            del self.values[:]
            del self.entropies[:]
            
            if(self.save):
                if(i_epoch%500==0):
                    PATH='./model/{}/A3C_{}.pkl'.format(self.level,self.level)
                    torch.save({
                                'epoch': i_epoch,
                                'model_state_dict': self.global_model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': total_loss,
                                }, PATH)
            if(i_epoch==80000):
                return
    def chooseAction(self,state):
        distribution=self.A(state)
        action=distribution.sample()
        log_prob=distribution.log_prob(action).unsqueeze(0)
        
        action=action.item()
        return action,log_prob


