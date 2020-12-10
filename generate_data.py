import os,json,ast, csv, torch, scipy.io, PIL.Image, torchvision.transforms
import numpy as np
import pandas as pd 

"""
Object: DF Index, IDx
Sky: 2,3
Tree: 4,5
Building: 1,2
Person: 12,13
Road: 6,7
Car: 20,21
Light: 82,83
"""
included_classes = {'Sky': 3, 'Tree':5, 'Building':2, 'Person':13,'Road':7,'Car':21,'Light':83}

def checkImage(path):
    labels = np.array(PIL.Image.open("./data/" + path))
    for idx in included_classes.values():
        if idx in labels:
            return True
    return False
def makeODGT(img_list,target_path):
    with open('./data/' + target_path,'w+') as f:
        for img_data in img_list:
            f.write(json.dumps(img_data) + "\n")   
            
with open('./data/training.odgt', 'r+') as f:
    train_data = f.readlines()
with open('./data/validation.odgt', 'r+') as f:
    val_data = f.readlines()    
limited_train_data = []
excluded = 0
for data in train_data:
    if checkImage(json.loads(data)['fpath_segm']):
        limited_train_data.append(json.loads(data))
    else:
        excluded+=1
#TEST
print((excluded+len(limited_train_data))==len(train_data))

limited_val_data = []
val_excluded = 0
for data in val_data:
    if checkImage(json.loads(data)['fpath_segm']):
        limited_val_data.append(json.loads(data))
    else:
        val_excluded+=1
#TEST
print((val_excluded+len(limited_val_data))==len(val_data))

makeODGT(limited_train_data,'limited_train.odgt')
makeODGT(limited_val_data,'limited_val.odgt')
