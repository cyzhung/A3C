# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:10:32 2021

@author: 123
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical

ACTIONS=12


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.conv1=nn.Conv2d(1,32,(3,3),stride=2, padding=1)
        self.conv2=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv3=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv4=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        
        self.lstm=nn.LSTMCell(32*5*5,512)
        #Actor full connect layer
        self.a_fc1=nn.Linear(512,ACTIONS)

        
        self.c_fc1=nn.Linear(512,1)


    def forward(self,x,hx,cx):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        
        hx,cx=self.lstm(x.view(x.size(0),-1),(hx,cx))
        
        policy=(self.a_fc1(hx))

        
        value=(self.c_fc1(cx))
        
        return policy,value,hx,cx
    
    
"""
class Actor(nn.Module):
     def __init__(self):
        super(Actor,self).__init__()
        self.conv1=nn.Conv2d(1,32,(3,3),stride=2, padding=1)
        self.conv2=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv3=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv4=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        
        #Actor full connect layer
        self.fc1=nn.Linear(32*5*5,512)
        self.fc2=nn.Linear(512,ACTIONS)
        #self.fc3=nn.Linear(64,ACTIONS)

        
        self.optimizer=optim.Adam(self.parameters(),lr=0.001)
        
     def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)  #flatten
        
        x=F.relu(self.fc1(x))
        x=(self.fc2(x))
       # x=(self.fc3(x))
        
        

        
        return x
     def learn(self,log_prob,tdError):
        
        loss=-tdError*log_prob
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss
    
     def updateParam(self,actor_loss):
        

        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer.step()
        
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.conv1=nn.Conv2d(1,32,(3,3),stride=2, padding=1)
        self.conv2=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv3=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        self.conv4=nn.Conv2d(32,32,(3,3),stride=2, padding=1)
        #Critic full connect layer
        self.fc1=nn.Linear(32*5*5,512)
        self.fc2=nn.Linear(512,1)
       # self.fc3=nn.Linear(32,1)

        self.GAMMA=0.99
        self.optimizer=optim.Adam(self.parameters(),lr=0.01)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))

        x = x.view(-1, 32 *5 * 5)  #flatten

        x=F.relu(self.fc1(x))
        x=(self.fc2(x))
       # x=(self.fc3(x))

        return x

    def learn(self,s,r,s_):
        v=self.forward(s)
        v_=self.forward(s_)
        print(v)
        print(v_)
        tdError=r+self.GAMMA*v_.detach()-v.detach()

        loss=torch.square(tdError)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return tdError,loss
    def updateParam(self,critic_loss):
        

        self.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer.step()

"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        