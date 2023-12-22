import pandas as pd
from preprocessing import get_data
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from temporal_averaging import temporal_avg

def train_model(data_load,test_data_loader,epochs,optimizer,criterion,model,tmp=False, diff=False, device=torch.device("cuda:1")):
    training_loss,training_accuracy,testing_loss,testing_accuracy=[],[],[],[]
    tloss_previous=10
    model.train()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        pred_cf=[]
        y_train=[]
        csamp,closs=0,0
        for i,(data,labels) in enumerate(tqdm(data_load)):
            if tmp:
                data,labels=temporal_avg(data, labels)
            data=data.to(device=device, dtype=torch.float)
            y_train.extend(labels.cpu().numpy())
            labels=labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            predictions=model(data)
            _,pred=torch.max(predictions,dim=1)
            pred_cf.extend(pred.cpu().numpy())
            csamp+=pred.eq(labels).sum().item()
            loss=criterion(predictions,labels)
            closs+=loss.item()
            loss.backward()
            optimizer.step()
        tloss, tacc,predictions_cf,y_test=test(test_data_loader,criterion,model,device,tmp,diff)
        if tloss<tloss_previous:
            cf_test=confusion_matrix(y_test,predictions_cf)
            cf_train=confusion_matrix(y_train,pred_cf)
        tloss_previous=tloss
        testing_accuracy.append(tacc)
        training_accuracy.append(csamp/len(data_load.dataset))
        testing_loss.append(tloss)
        training_loss.append(closs/len(data_load))
    history={}
    history['training_loss']=training_loss
    history['training_accuracy']=training_accuracy
    history['validation_loss']=testing_loss
    history['validation_accuracy']=testing_accuracy
    history['cf_test']=cf_test
    history['cf_train']=cf_train
    
    return history

def test(data_load, criterion, model, device, tmp, diff):
    predictions_cf,y_test=[],[]
    model.eval()
    csamp,closs=0,0
    with torch.no_grad():
        for i,(data,labels) in enumerate(tqdm(data_load)):
            if tmp:
                data, labels=temporal_avg(data, labels)
            y_test.extend(labels.cpu().numpy())
            data=data.to(device=device, dtype=torch.float)
            labels=labels.to(device=device, dtype=torch.long)
            predictions=model(data)
            _,pred=torch.max(predictions,dim=1)
            predictions_cf.extend(pred.cpu().numpy())
            csamp+=pred.eq(labels).sum().item()
            loss=criterion(predictions,labels)
            closs+=loss.item()
    return closs/len(data_load),csamp/len(data_load.dataset),predictions_cf, y_test 
