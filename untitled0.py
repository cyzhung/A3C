# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:44:21 2021

@author: 123
"""
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def load_data():
    train_data=pd.read_csv("./train.csv")
    label=(train_data[["11"]])
    label=label.to_numpy()

    train_data.drop(['1','11'],inplace=True,axis=1)
    train_data=train_data.to_numpy()

    return train_data,label
    

if __name__=='__main__':
    train_data,label=load_data()
    X_train,X_test, y_train, y_test =train_test_split(train_data,label,test_size=0.2,random_state=10)
    model=LinearRegression()
    model.fit(X_train,y_train)
    
    print("決定係數(train):{:.3f}".format(model.score(X_train,y_train)))
    print("決定係數(train):{:.3f}".format(model.score(X_test,y_test)))