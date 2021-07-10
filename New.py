from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape  
from keras.layers.convolutional import MaxPooling2D  
from keras.layers import Conv2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop

import os
import random
import numpy as np 
from PIL import Image

import datetime

def load_data(dir1,stride):
    imgs=os.listdir(dir1)
    num=int(len(imgs)/stride)
    data=np.empty((num,stride,75,150,1),dtype="float16")
    label=np.empty((num,2),dtype="float16")
    
    flag=0
    for i in range(num):                              
        while flag<stride:
            img=Image.open(dir1+'\\'+imgs[i*stride+flag])	
            arr=np.asarray(img,dtype="float32")
            data[i,flag,:,:,:]=arr[:,:,None]
            label[i,0]=int(imgs[i*stride+flag].split('-')[0])/100-2
            label[i,1]=int(imgs[i*stride+flag].split('-')[1])/100-4
            flag=flag+1
        flag=0
    return data,label,num

stride=11
np.random.seed(stride)


dir1='G:\\搅拌机视频\\2.等级\\C30\\4.预处理\\8.合并'
data,label,num = load_data(dir1,stride) 
data=data/255


Ndata=np.empty((num*stride,stride,75,150,1),dtype="float16")
Nlabel=np.empty((num*stride,2),dtype="float16")
for i in range(num):
    for j in range(stride):        
        for nn in range(stride):
            if j+nn>stride-1:
                Ndata[i*stride+j,nn]=data[i,nn+j-stride]               
            else:
                Ndata[i*stride+j,nn]=data[i,nn+j]       
        Nlabel[i*stride+j]=label[i]                

index = [i for i in range(len(Ndata))]
random.shuffle(index)
Ndata = Ndata[index]
Nlabel = Nlabel[index]


model = Sequential()

model.add(TimeDistributed(Conv2D(4, (2, 2), padding="valid"), input_shape=(stride,75,150,1)))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Conv2D(4, (2, 2), padding="valid")))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Conv2D(4, (2, 2), padding="valid")))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(LSTM(return_sequences=False, units=5))
model.add(Dense(2,activation='linear'))

rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rmsprop,metrics=['accuracy'])
print (model.summary())

begin=datetime.datetime.now()
model.fit(Ndata, Nlabel, batch_size=22, epochs=15,validation_split=0.2) 
end=datetime.datetime.now()
print (end-begin)
json_string=model.to_json()
open('cc-202107091558.json','w').write(json_string)    
model.save_weights('cc-202107091558.h5')    

