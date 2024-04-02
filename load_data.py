import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

#%matplotlib inline

mnist = fetch_openml("mnist_784",version=1,parser='auto')
train_images, train_labels = mnist.data,mnist.target
print(train_images.shape,train_labels.shape)
df = train_images
df['Class'] = train_labels
df.head()
df_even = df[(df["Class"]).astype(int)%2==0]
df_odd = df[(df["Class"]).astype(int)%2!=0]
df_even.to_csv("df_even.csv",index=False)
df_odd.to_csv("df_odd.csv",index=False)