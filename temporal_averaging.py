import torch
import numpy as np

def temporal_avg(data, labels):
    data, labels=data.numpy(), labels.numpy()
    ind_attack=[i for i in range(len(labels)) if list(labels)[i]==0]
    ind_benign=[i for i in range(len(labels)) if list(labels)[i]==1]
    
    data_att=[np.array(data[i]) for i in ind_attack]
    data_ben=[np.array(data[i]) for i in ind_benign]
    
    label_att=[labels[i] for i in ind_attack]
    label_ben=[labels[i] for i in ind_benign]
    
    attack, labels_attack = get_averaged_data(data_att, label_att)
    benign, labels_benign = get_averaged_data(data_ben, label_ben)
    
    attack=attack+benign
    labels_attack=labels_attack+labels_benign

    return torch.FloatTensor(attack), torch.FloatTensor(labels_attack)

def get_averaged_data(data, labels):
    n=3 if len(labels)>3 else len(labels)
    lst=[]
    data_avg, labels_avg=[],[]
    for i in range(len(data)):
        lst=[data[j] if j<len(data) else data[j-len(data)] for j in range(i,i+n)]
        data_avg.append(list(sum(lst)/n))
        labels_avg.append(labels[i])
    return data_avg, labels_avg

def diff_input(data,labels):
    print(labels.unique())
    data, labels=data.numpy(), labels.numpy()
    ind_attack=[i for i in range(len(labels)) if list(labels)[i]==0]
    ind_benign=[i for i in range(len(labels)) if list(labels)[i]==1]
    print('indices', len(ind_attack), len(ind_benign)) 
    data_att=[np.array(data[i]) for i in ind_attack]
    data_ben=[np.array(data[i]) for i in ind_benign]
    
    label_att=[labels[i] for i in ind_attack]
    label_ben=[labels[i] for i in ind_benign]
    
    attack, labels_attack=get_diff_data(data_att, label_att)
    benign, labels_benign=get_diff_data(data_ben, label_ben)
    print('labels ben labels att', len(label_att), len(label_ben))
    attack=attack+benign
    labels_attack=labels_attack+labels_benign
    print(len(labels_attack))
    
    return torch.FloatTensor(attack), torch.FloatTensor(labels_attack)
    
def get_diff_data(data, labels):
    data_avg, labels_avg=[],[]
    if len(data):
        data_avg.append(list(data[0]))
        labels_avg.append(labels[0])
    
    for i in range(1,len(data)):
        data_avg.append(data[i]-data[i-1])
        labels_avg.append(labels[i])
    return data_avg, labels_avg

