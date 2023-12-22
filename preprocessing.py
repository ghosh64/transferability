import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Encode:
    def __init__(self):
        self.classes={'Normal':['normal'],
                      'Dos':['neptune','teardrop','smurf','pod','back','land','apache2','processtable','mailbomb','udpstorm'],
                      'U2R':['rootkit','buffer_overflow','loadmodule','perl','httptunnel','ps','xterm','sqlattack'],
                      'R2L':['warezclient','ftp_write','phf','multihop','guess_passwd','warezmaster','spy','imap','snmpgetattack','snmpguess','multihop','named','sendmail','worm','xlock','xsnoop'],
                      'Probing':['ipsweep','portsweep', 'nmap','satan','saint','mscan']}
        self.keys=list(self.classes.keys())
    
    def getlabels(self,labels):
        converted_labels=np.zeros(len(labels))
        for i,label in enumerate(labels):
            converted_labels[i]=[self.keys.index(k) for k in self.classes.keys() if (label in self.classes[k])][0]
        return converted_labels
        
    def __call__(self,labels):
        return self.getlabels(labels)
    
def get_labels(labels):
    encode=Encode()
    return encode(labels)

def text_to_int(data_train):
    columns=[col for col in data_train.columns if data_train[col].dtype=='object']
    le=LabelEncoder()
    for col in columns:
        data_train[col]=le.fit_transform(data_train[col])
    return data_train

def data_scaler(data_train):
    minmaxscaler=MinMaxScaler()
    for col in data_train.columns:
        data_train[col]=minmaxscaler.fit_transform(np.array(data_train[col]).reshape(-1,1))
    return data_train

def get_data(datasets):
    data, labels=dict(),dict()
    for k in datasets.keys():
        dataset=datasets[k]
        data[k]=dataset.iloc[:,:41]
    
        #convert text columns to int
        data[k]=text_to_int(data[k])
    
        #scale data --> [0,1]
        data[k]=data_scaler(data[k])
        
        #label encoded labels
        label=dataset[41]
        labels[k]=get_labels(label)
    
    return np.array(data[0]), labels[0], np.array(data[1]), labels[1]