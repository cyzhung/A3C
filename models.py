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
        
        self.fc=nn.Linear(32*5*5,512)

        #Actor full connect layer
        self.a_fc1=nn.Linear(512,ACTIONS)

        
        self.c_fc1=nn.Linear(512,1)


    def forward(self,x,hx,cx):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        
        #hx,cx=self.lstm(x.view(x.size(0),-1),(hx,cx))
        x=x.view(-1,32*5*5)
        x=F.relu(self.fc(x))

        policy=(self.a_fc1(x))


        
        value=(self.c_fc1(x))

        
        return policy,value,hx,cx



class ActorCritic_LSTM(nn.Module):
    def __init__(self):
        super(ActorCritic_LSTM,self).__init__()
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




        
        
        
        
        
        
        
        
        
        
        
        
        
        