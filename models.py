import torch.nn as nn
import torch
from torch.nn import functional as F

class cldnn(nn.Module):
    def __init__(self, n_classes=12):
        super().__init__()
        self.layer1=nn.Conv2d(1,256,(1,3))
        self.layer2=nn.Conv2d(256,256,(2,3))
        self.leakyrelu=nn.LeakyReLU(0.3)
        self.layer3=nn.Conv2d(256,256,(1,3))
        self.dropout=nn.Dropout(0.2)
        self.layer4=nn.Conv2d(256,80,(1,3))
        self.flatten=nn.Flatten()
        self.linear=nn.Linear(2480,128)
        self.output=nn.Linear(128,n_classes)
        self.sigmoid=nn.Softmax()
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=self.layer2(x)
        x=self.leakyrelu(x)
        x=self.layer3(x)
        x=self.dropout(x)
        x=F.relu(self.layer4(x))
        x=torch.reshape(x,(-1,31,80))
        x=self.flatten(x)
        #print(x.shape)
        x=F.relu(self.linear(x))
        x=self.output(x)
        #change to softmax?
        #x=self.sigmoid(x)
        return x
        

'''def cldnn():
# FILL THIS IN WITH MODEL ARCHITECTURE
model = Sequential()
model.add(Conv2D(256,(1,3),activation="relu",input_shape =(2,39,1)))
model.add(Conv2D(256,(2,3),activation=None))
model.add(LeakyReLU(0.3))
model.add(Conv2D(256,(1,3),activation=None))
model.add(LeakyReLU(0.3))
model.add(Dropout(0.20))
model.add(Conv2D(80,(1,3),activation="relu"))
model.add(Reshape((31,80)))
model.add(Flatten())
model.add(Dense(128,activation="relu",kernel_initializer="normal"))
model.add(Dense(2,activation="sigmoid"))
return model'''