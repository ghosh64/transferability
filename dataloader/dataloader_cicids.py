from torch.utils.data import Dataset
import os
#import json
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms
import torch
#from sklearn.neighbors import NearestNeighbors
from random import choice
import math

class dataset_cicids(Dataset):
    def __init__(self,
                 device, 
                 path='datasets/',
                 dataset_loc='../',
                 train=False,
                 test=False,
                 val=False,
                 two_class=False,
                 bootstrap=False,
                 synthetic=False,
                 separate_nodes=False,
                 transform=transforms.ToTensor()):
        self.bootstrap=bootstrap
        self.synthetic=synthetic
        self.separate_nodes=separate_nodes
        self.path=path+'separate_nodes/bootstrap/' if (self.separate_nodes and self.bootstrap) else path+'separate_nodes/synthetic/' if (self.separate_nodes and self.synthetic) else path+'separate_nodes/normal/' if self.separate_nodes else path+'bootstrap/' if self.bootstrap else path+'synthetic/' if self.synthetic else path+'normal/'
        #print(self.path)
        self.device=device
        self.n_devices=5
        self.transform=transform
        self.dataset_loc=dataset_loc
        self.two_class=two_class if not self.separate_nodes else True
        self.mode='train' if train else 'val' if val else 'test'
        if self.separate_nodes:self.n_devices=11
            #self.create_separate_datasets()
        if not os.path.exists(self.path+f'dataset_{self.device}.txt'):
            #create dataset here
            if not os.path.exists(self.path): os.makedirs(self.path)
            if self.separate_nodes:
                #for this to work you need to set the synthetic or bootstrap for those two options
                #for the normal one, you don't need to set it
                if self.synthetic:
                    self.create_separate_datasets_synthetic()  
                else: self.create_separate_datasets()
            else:
                self.create_dataset()
                self.create_train_val_test_split()
        if not os.path.exists(self.path+f'device_{self.device}/'+f'dataset_{self.mode}.txt'):
            if not os.path.exists(self.path+f'device_{self.device}/'): os.makedirs(self.path+f'device_{self.device}/')
            self.create_train_val_test_split()
        with open(self.path+f'device_{self.device}/'+f'dataset_{self.mode}.txt','rb') as f:a=pickle.load(f)
        self.data, self.labels, self.classes=a['dataset'],a['labels'],a['classes']
        
    def create_separate_datasets_synthetic(self):
        dataset=self.read_csvs()
        x_dataset,y=self.clean_dataset(dataset)
        y_encoded=self.labelencode(y)
        x_dataset_scaled=self.scale_data(x_dataset)
        y_attacks=self.get_eligible_attacks(y_encoded)
        datasets=[pd.DataFrame() for _ in range(self.n_devices)]
        labels=[[] for _ in range(self.n_devices)]
        y_encoded=np.array(y_encoded)
        benign_indices=[[] for _ in range(self.n_devices)]
        frac=1/self.n_devices
        y_attacks=np.array(y_attacks)
        y_encoded=np.array(y_encoded)
        for device in range(self.n_devices):
            required_attacks=[0, y_attacks[device+1]]
            print(required_attacks)
            for attack in required_attacks:
                indices=np.where(y_encoded==attack)[0]
                if attack==0:
                    indices_device=indices[(int)(device*frac*len(indices)):(int)((device+1)*frac*len(indices))]
                    x=x_dataset_scaled[indices_device]
                    y=y_encoded[indices_device]
                if attack>0:
                    #print(attack)
                    n=math.ceil(len(indices_device)/len(indices))
                    x_attack, y_attack=x_dataset_scaled[indices],y_encoded[indices]
                    x_attack_smote=self.SMOTE(x_attack, 100,5)
                    #print(len(x), len(x_attack_smote))
                    #print(n)
                    if n-1>0:
                        x=np.concatenate([x,np.vstack([x_attack_smote]*(n-1))])
                        y=np.concatenate([y,np.tile(y_attack, n-1)])
                    else: 
                        x=np.concatenate([x,x_attack_smote])
                        y=np.concatenate([y,y_attack])
                a={}
                a['dataset']=pd.DataFrame(x)
                a['labels']=y
                a['classes']=self.classes
            
                with open(self.path+f'dataset_{device}.txt','wb') as f: pickle.dump(a,f)
                if not os.path.exists(self.path+f'device_{device}/'): os.makedirs(self.path+f'device_{device}/')
                self.create_train_val_test_split(device)
        return
        
    def create_separate_datasets(self):
        dataset=self.read_csvs()
        x_dataset,y=self.clean_dataset(dataset)
        y_encoded=self.labelencode(y)
        x_dataset_scaled=self.scale_data(x_dataset)
        y_attacks=self.get_eligible_attacks(y_encoded)
        datasets=[pd.DataFrame() for _ in range(self.n_devices)]
        labels=[[] for _ in range(self.n_devices)]
        y_encoded=np.array(y_encoded)
        benign_indices=[[] for _ in range(self.n_devices)]
        frac=1/self.n_devices
        y_attacks=np.array(y_attacks)
        y_encoded=np.array(y_encoded)
        if self.bootstrap: ben_len=[]
        for attack in y_attacks:
            indices=np.where(y_encoded==attack)[0]
            if attack==0:
                for device in range(self.n_devices):
                    benign_indices[device].extend(indices[(int)(device*frac*len(indices)):(int)((device+1)*frac*len(indices))])
                    if self.bootstrap: ben_len.append(len(benign_indices[device]))
            if attack>0:
                benign_indices[np.where(y_attacks==attack)[0][0]-1].extend(indices)
                if self.bootstrap:
                    n=math.ceil(ben_len[np.where(y_attacks==attack)[0][0]-1]/len(indices))
                    for _ in range(n-1):benign_indices[np.where(y_attacks==attack)[0][0]-1].extend(indices)
                #if self.synthetic:     
        for device in range(self.n_devices):
            dataset=x_dataset_scaled[np.array(benign_indices[device])]
            labels=y_encoded[np.array(benign_indices[device])]
            a={}
            a['dataset']=pd.DataFrame(dataset)
            a['labels']=labels
            #since the labels are not going to be exact, we can keep the original self.classes
            #also each dataset has one non zero class and to see what class it is, you can just do self.classes[index]
            #since we did not do exact labels for the labels, we can directly index. self.classes_actual is not usable since
            #we would need to do exact labels for labels.
            a['classes']=self.classes
            with open(self.path+f'dataset_{device}.txt','wb') as f: pickle.dump(a,f)
            if not os.path.exists(self.path+f'device_{device}/'): os.makedirs(self.path+f'device_{device}/')
            #print('device',device)
            self.create_train_val_test_split(device)
        return 
        
    def create_train_val_test_split(self,device=-1):
        dev=self.device if device==-1 else device
        #print(dev)
        with open(self.path+f'dataset_{dev}.txt','rb') as f:a=pickle.load(f)
        dataset_local, labels_local, classes_exact=a['dataset'],a['labels'],a['classes']
        #print(len(dataset_local))
        labels_local=np.array(labels_local)
        modes={'train':0.8,'val':0.1,'test': 0.1}
        indices_dict={'train':[],'test':[],'val':[]}
        
        for attack in np.unique(labels_local):
            frac=0
            indices=np.where(labels_local==attack)[0]
            for mode in modes.keys():
                indices_mode=indices[frac:int(frac+(modes[mode]*len(indices)))]
                indices_dict[mode].extend(indices_mode)
                frac+=int(modes[mode]*len(indices))
                
        for mode in modes.keys():
            ind=np.array(indices_dict[mode])
            dataset=dataset_local.loc[ind,:]
            labels=labels_local[ind]
           
            a={}
            a['dataset']=dataset
            a['labels']=labels
            a['classes']=classes_exact
            if not os.path.exists(self.path+f'device_{dev}/'): os.makedirs(self.path+f'device_{dev}')
            with open(self.path+f'device_{dev}/'+f'dataset_{mode}.txt','wb') as f: pickle.dump(a,f)
        return 
        
        
    def read_csvs(self):
        dataset=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv')
        dataset_1=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv')
        dataset_2=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
        dataset_3=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
        dataset_4=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
        dataset_5=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
        dataset_6=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv')
        dataset_7=pd.read_csv(f'{self.dataset_loc}'+'MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv')
        
        #consolidate csvs here
        dataset=dataset.append(dataset_1, ignore_index=True)
        dataset=dataset.append(dataset_2, ignore_index=True)
        dataset=dataset.append(dataset_3, ignore_index=True)
        dataset=dataset.append(dataset_4, ignore_index=True)
        dataset=dataset.append(dataset_5, ignore_index=True)
        dataset=dataset.append(dataset_6, ignore_index=True)
        dataset=dataset.append(dataset_7, ignore_index=True)
        
        #print(len(dataset))
        return dataset
    
    def clean_dataset(self,dataset):
        #dataset.head()
        dataset[dataset==np.inf]=np.nan
        dataset.fillna(0,inplace=True)
        x_dataset=dataset.iloc[:, :-1]
        y=dataset.iloc[:, -1]
        
        return x_dataset,y
    
    def labelencode(self,y):
        le=LabelEncoder()
        y_encoded=le.fit_transform(y)
        
        self.classes=le.classes_
        return y_encoded
    
    def scale_data(self,x_dataset):
        scaler=MinMaxScaler()
        x_dataset_scaled=scaler.fit_transform(x_dataset)
        
        return x_dataset_scaled
    
    def get_eligible_attacks(self,y_encoded):
        y_val,y_count = np.unique(y_encoded, return_counts=True) 
        y_retained = y_val[y_count>100]
        #y_attacks = y_retained[y_retained>0]
        
        return y_retained
    def get_exact_labels(self, labels,y_attacks):
        for device in range(self.n_devices):
            labels[device]=[np.where(y_attacks==labels[device][i])[0][0] for i in range(len(labels[device]))]
        return labels
    def get_exact_labels_one_device(self, labels,y_attacks):
        #print(np.array(y_attacks))
        #print(np.unique(labels))
        labels=[np.where(y_attacks==labels[i])[0][0] for i in range(len(labels))]
        return labels
    
    def save_bootstrapped_data(self, bootstrapped_indices, y_encoded, x_dataset_scaled,y_attacks):
        for device in range(self.n_devices):
            dataset=x_dataset_scaled[np.array(bootstrapped_indices[device])]
            labels=y_encoded[np.array(bootstrapped_indices[device])]
            labels=self.get_exact_labels_one_device(labels, y_attacks)
            a={}
            a['dataset']=pd.DataFrame(dataset)
            a['labels']=labels
            a['classes']=self.classes_actual
            with open(self.path+f'dataset_{device}.txt','wb') as f: pickle.dump(a,f)
        return
            
    def save_synthetic_data(self, y_encoded, x_dataset_scaled,y_attacks):
        frac=1/self.n_devices
        for device in range(self.n_devices):
            dataset_device, label_device=np.array([]),np.array([])
            for attack in y_attacks:
                indices=np.where(y_encoded==attack)[0]
                indices_device=indices[(int)(device*frac*len(indices)):(int)((device+1)*frac*len(indices))]
                y_device=y_encoded[indices_device]
                x_device=x_dataset_scaled[indices_device]
                if attack==0: 
                    len_benign=len(indices_device)
                    dataset_device=x_device
                    label_device=y_device
                if attack>0:
                    x_attack_train_smote=self.SMOTE(x_device, 100,5)
                    y_attack_train_smote=np.array([y_device[0] for _ in range(len(x_device))])
                    n=int(len_benign/len(y_attack_train_smote))
                    dataset_device=np.concatenate([dataset_device, np.vstack([x_attack_train_smote]*(n-1))])
                    label_device=np.concatenate([label_device, np.tile(y_attack_train_smote,n-1)])
            label_device=self.get_exact_labels_one_device(label_device, y_attacks)
            a={}
            a['dataset']=pd.DataFrame(dataset_device)
            a['labels']=label_device
            a['classes']=self.classes_actual
            with open(self.path+f'dataset_{device}.txt','wb') as f: pickle.dump(a,f)
                
        return    
    
    def SMOTE(self,T, N, k):
        # """
        # Returns (N/100) * n_minority_samples synthetic minority samples.
        #
        # Parameters
        # ----------
        # T : array-like, shape = [n_minority_samples, n_features]
        #     Holds the minority samples
        # N : percetange of new synthetic samples:
        #     n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        # k : int. Number of nearest neighbours.
        #
        # Returns
        # -------
        # S : array, shape = [(N/100) * n_minority_samples, n_features]
        # """
        n_minority_samples, n_features = T.shape

        if N < 100:
            #create synthetic samples only for a subset of T.
            #TODO: select random minortiy samples
            N = 100
            pass

        if (N % 100) != 0:
            raise ValueError("N must be < 100 or multiple of 100")

        N = int(N/100)
        n_synthetic_samples = N * n_minority_samples
        n_synthetic_samples = int(n_synthetic_samples)
        n_features = int(n_features)
        S = np.zeros(shape=(n_synthetic_samples, n_features))

        #Learn nearest neighbours
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(T)

        #Calculate synthetic samples
        for i in range(n_minority_samples):
            nn = neigh.kneighbors(T[i].reshape(1, -1), return_distance=False)
            for n in range(N):
                nn_index = choice(nn[0])
                #NOTE: nn includes T[i], we don't want to select it
                while nn_index == i:
                    nn_index = choice(nn[0])

                dif = T[nn_index] - T[i]
                gap = np.random.random()
                S[n + i * N, :] = T[i,:] + gap * dif[:]

        return S
    
    def create_dataset(self):
        dataset=self.read_csvs()
        x_dataset,y=self.clean_dataset(dataset)
        y_encoded=self.labelencode(y)
        x_dataset_scaled=self.scale_data(x_dataset)
        #We only consider the classes that have atleast 100 data points in them
        y_attacks=self.get_eligible_attacks(y_encoded)
        
        self.classes_actual=[self.classes[cls] for cls in y_attacks]

        #print(np.unique(self.labels))
        #print(self.classes_actual)
        #when you are creating the dataset, please make sure you save self.classes to see what the classes are
        #also since the classes will be missing, you need to do np.unique() to get unique classes and then 
        #create class labels from indexing np.unique. to get original classes, you would need to index np.unique()
        #array with the class label to get the original class number and then use the obtained class number as an
        #index to self.classes to get the actual name of the class.
        
        #the other thing is that you can create an array of the classes based on y_attacks and then index using
        #np.unique because it would order them the same way and then you can directly look up the class name
        #from the index and the self.classes_actual
        
        datasets=[pd.DataFrame() for _ in range(self.n_devices)]
        labels=[[] for _ in range(self.n_devices)]
        y_encoded=np.array(y_encoded)
        if self.synthetic:self.save_synthetic_data(y_encoded, x_dataset_scaled,y_attacks)
        if self.bootstrap: 
            bootstrapped_indices=[[] for _ in range(self.n_devices)]
            length_benign=[]
        if not self.synthetic:
            for attack in y_attacks:
                indices=np.where(y_encoded==attack)[0]
                frac=1/self.n_devices
                for i in range(self.n_devices):
                    indices_device=indices[(int)(i*frac*len(indices)):(int)((i+1)*frac*len(indices))]
                    if self.bootstrap:
                        #stores indices because storing bootstrapped data of 5 was causing the kernel to die
                        bootstrapped_indices[i].extend(indices_device)
                        if attack==0: 
                            length_benign.append(len(indices_device))
                        if attack>0: 
                            n=math.ceil(length_benign[i]/len(indices_device))
                            for _ in range(n-1):bootstrapped_indices[i].extend(indices_device)
                        continue
                    y_device=y_encoded[indices_device]
                    x_device=x_dataset_scaled[indices_device]
                    x_device=pd.DataFrame(x_device)
                    datasets[i]=datasets[i].append(x_device,ignore_index=True)
                    labels[i].extend(y_device)
        if self.bootstrap:
            self.save_bootstrapped_data(bootstrapped_indices,y_encoded, x_dataset_scaled,y_attacks)
        if not (self.bootstrap or self.synthetic):
            labels=self.get_exact_labels(labels,y_attacks)
            for device in range(self.n_devices):
                a={}
                a['dataset']=datasets[device]
                a['labels']=labels[device]
                a['classes']=self.classes_actual
                with open(self.path+f'dataset_{device}.txt','wb') as f: pickle.dump(a,f)
                
        return
    
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        data,label=np.array(self.data.iloc[idx]),self.labels[idx]
        data=data.reshape(2,39,1)
        if self.two_class: label=1 if label>0 else label
        return self.transform(data), label
        
