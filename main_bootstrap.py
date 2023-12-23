import pandas as pd
from preprocessing import get_data
import numpy as np
import torch
import torch.optim as optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
#from dataloader.dataloader_nsskdd import dataset_whole
#from dataloader.dataloader_federated import dataset,dataset_transfer
from dataloader.dataloader_cicids import dataset_cicids
from models import cldnn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from train_func import train_model, test
from aggregation_algorithms import FedAvg
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--model',type=str,default='cldnn')
parser.add_argument('--n_classes', type=int, default=15)
parser.add_argument('--algo', type=str, default='FedAvg')
parser.add_argument('--device',type=int, default=0)
#parser.add_argument('--silobn',type=str, default='False')
parser.add_argument('--weight_path',type=str,default='weights/')
parser.add_argument('--tb_path', type=str, default='tensorboard/')
parser.add_argument('--transferability', type=str, default='False')
parser.add_argument('--cf',type=str,default='conf_matrix_figures/')
parser.add_argument('--n_rounds', type=int, default=20)
parser.add_argument('--tmp', type=str, default='False')
parser.add_argument('--diff', type=str, default='False')
parser.add_argument('--epochs',type=int, default=20)
args=parser.parse_args()

#transferability/multi-class
if args.transferability=='True': 
    args.transferability=True
    args.n_classes=2
else: args.transferability=False

device=torch.device(f"cuda:{args.device}")
args.model=args.model+'(args.n_classes)'

if not os.path.exists(args.cf):
    os.makedirs(args.cf)

if not os.path.exists(args.tb_path):
    os.makedirs(args.tb_path)
if not os.path.exists(args.weight_path):
    os.makedirs(args.weight_path)

tb=SummaryWriter(f'{args.tb_path}')

n_rounds=args.n_rounds

n_devices=5 if not args.transferability else 11

#check what kind of model is chosen
global_model=eval(args.model)
two_class=False
#change n_classes here for transferability
#if args.n_classes==2 or args.transferability: 
#    two_class=True

tmp, diff=True if args.tmp=='True' else False, False

state_dict={}

#Get length of dataset of each device for the fedavg algorithm
dataset_len=np.zeros(n_devices)

for i in range(n_devices):
    if args.transferability:
        training_data=dataset_cicids(device=i, train=True,bootstrap=True,separate_nodes=True)
    else: training_data=dataset_cicids(device=i,train=True,bootstrap=True)
    dataset_len[i]=len(training_data.labels)

path=args.weight_path

torch.save(global_model.state_dict(),path+'global_model_weights.pt')
test_loss, test_acc=[],[]
for comm_round in range(n_rounds):
    accuracies=np.zeros(n_devices)
    losses=np.zeros(n_devices)
    print(f"********** Communication Round {comm_round+1}******************")
    for dev in range(n_devices):
        print(f'.............Device {dev + 1}.................')
        #get local training data
        #get local testing/validation data
        if not args.transferability:
            training_data=dataset_cicids(device=dev,train=True,bootstrap=True)
            val_data=dataset_cicids(device=dev,val=True,bootstrap=True)
        else:
            training_data=dataset_cicids(device=dev, train=True, bootstrap=True,separate_nodes=True)
            val_data=dataset_cicids(device=dev,val=True, bootstrap=True, separate_nodes=True)
        train_data_loader=DataLoader(training_data, batch_size=1024, shuffle=True)
        val_data_loader=DataLoader(val_data, batch_size=1024, shuffle=True)
        
        #initialize models with global model
        #check what kind of model chosen
        device_model=eval(args.model)
        sd=torch.load(path+'global_model_weights.pt')
                  
        device_model.load_state_dict(sd)
        #optimizer, loss
        optimizer_parameters=device_model.parameters()
        optim=optimizer.Adam(optimizer_parameters,lr=0.0001, weight_decay=1e-2)
        criterion=nn.CrossEntropyLoss()
        epochs=args.epochs
        
        #train model on local data
        #history: training_accuracy, training_loss, validation_accuracy, validation_loss, cf_training, cf_testing
        history=train_model(train_data_loader, val_data_loader, epochs, optim, criterion, device_model, tmp,diff,device=device)
        
        #add device validation accuracy and validation loss to tensorboard
        #validation accuracy and validation loss in tb is the average of the accuracies/losses over 20 training epochs
        tb.add_scalar(f'Device {dev}: Training accuracy', sum(history['training_accuracy'])/len(history['training_accuracy']),comm_round)
        tb.add_scalar(f'Device {dev}: Training loss', sum(history['training_loss'])/len(history['training_loss']),comm_round)
        tb.add_scalar(f'Device {dev}: Validation accuracy', sum(history['validation_accuracy'])/len(history['validation_accuracy']),comm_round)
        tb.add_scalar(f'Device {dev}: Validation loss', sum(history['validation_loss'])/len(history['validation_loss']),comm_round)
        
        #store the trained weights in a dict
        state_dict[dev]=device_model.state_dict()
        
        #test device on any testing set other than it's because it is being validated on it
        #testing_set=np.arange(0,n_devices,1)
        
        if args.transferability:
            test_data=dataset_cicids(device=dev, test=True, bootstrap=True,separate_nodes=True)
        else:test_data=dataset_cicids(device=dev, test=True, bootstrap=True)
        test_data_loader=DataLoader(test_data, batch_size=1024, shuffle=True)
       
        #if you want to test on original data, remove tmp, diff from test()
        testing_loss, testing_accuracy,_,_=test(test_data_loader,criterion, device_model,device,tmp,diff)
        accuracies[dev]=testing_accuracy
        losses[dev]=testing_loss
        
        #log tensorboard accuracies and loss for each device
        tb.add_scalar(f'Device {dev}:Testing accuracy', testing_accuracy,comm_round)
        tb.add_scalar(f'Device {dev}:Testing loss', testing_loss,comm_round)

        if (comm_round+1)%2==0:
            testing_set=np.arange(0,n_devices,1)
            if args.transferability: cfs=[]
            for testing_device in testing_set:
                dataset_test=dataset_cicids(device=testing_device,test=True, bootstrap=True) if not args.transferability else dataset_cicids(device=testing_device, test=True, bootstrap=True, separate_nodes=True)
                data_load=DataLoader(dataset_test,batch_size=1024,shuffle=True)
                #if you want to test on original data, remove tmp, diff from test()
                loss,accuracy,y_pred,y_test=test(data_load,criterion,device_model,device,tmp,diff)
                cf_test=confusion_matrix(y_test,y_pred)
                if args.transferability : cfs.append(cf_test)
                else:
                    disp=ConfusionMatrixDisplay(confusion_matrix=cf_test)
                    disp.plot()
                    cf_path=args.cf+f'comm_round_{comm_round+1}/device_{dev}/'
                
                    if not os.path.exists(cf_path):
                        os.makedirs(cf_path)
                
                    disp.figure_.savefig(cf_path+f'cf_{testing_device}.png',dpi=300)
            if args.transferability:    
                cf_path=args.cf+f'comm_round_{comm_round+1}/device_{dev}/'
                if not os.path.exists(cf_path):os.makedirs(cf_path)
                with open(cf_path+f'dataset_cfs.txt','wb') as f: pickle.dump(cfs,f)
    
    print(accuracies, accuracies.mean())
    #choose aggregation algorithm: FedAvg
    updated_weights=FedAvg(state_dict,n_devices,dataset_len,device)
    
    #update global weights
    torch.save(updated_weights,path+'global_model_weights.pt')
    #save weights for each comm round
    if not os.path.exists(path+f'comm_round_{comm_round+1}/'): os.makedirs(path+f'comm_round_{comm_round+1}/')
    torch.save(updated_weights, path+f'comm_round_{comm_round+1}/global_model_weights.pt')
    
    tb.add_scalar('Average Testing loss', losses.mean(), comm_round)
    tb.add_scalar('Average Testing accuracy', accuracies.mean(), comm_round)
    
    test_loss.append(losses.mean())
    test_acc.append(accuracies.mean())

tb.close()
