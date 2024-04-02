import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Softmax_Regression import Softmax
from Neural_Network import NeuralNetwork
import Neural_Network as neu

user = input("Choose 1 for Sotmax Regression or 2 for Neural Network: ")
if user == "1":
    user1 = input("Choos 1 for df_even or 2 for df_odd: ")
    if user1 == "1":
        model_s = Softmax(784,5)
        X, y = model_s.split(model_s.load_data("df_even"))
        X_train, X_test, y_train, y_test = train_test_split(X, model_s.encode_even(y), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)/255
        model_s.fit_foftmax(X_train_tensor,y_train_tensor,X_test_tensor,y_test,model_s,300,0.01)
    else:
        model_s = Softmax(784,5)
        X, y = model_s.split(model_s.load_data("df_odd"))
        X_train, X_test, y_train, y_test = train_test_split(X, model_s.encode_odd(y), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)/255
        model_s.fit_foftmax(X_train_tensor,y_train_tensor,X_test_tensor,y_test,model_s,300,0.01)
else:
    user2 = input("Choose 1 for for df_odd to df_even and 2 for df_even to df_odd")
    if user2 == "1":
        input_size = 784
        hidden_size = 500
        num_classes = 5
        #learning_rate = 0.01
        model = NeuralNetwork(input_size, hidden_size, num_classes)

        X1, y1 = model.split(model.load_data("df_odd"))
        X_train, X_test, y_train, y_test = train_test_split(X1, model.encode_odd(y1), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)/255
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        neu.fit_neural_network(X_train_tensor,y_train_tensor,X_test_tensor,y_test,model)
        
        print("Before freezing:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)


        for param in model.fc1.parameters():
            param.requires_grad = False
        #Transfer Learning training    
        num_ftrs = model.fc2.in_features
        model.fc2 = nn.Linear(num_ftrs, 5)

        X2, y2 = model.split(model.load_data("df_even"))
        X_train, X_test, y_train, y_test = train_test_split(X2, model.encode_even(y2), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)

        X_train_tensor1 = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor1 = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor1 = torch.tensor(X_test, dtype=torch.float32)/255
        y_test_tensor1 = torch.tensor(y_test, dtype=torch.float32)

        neu.fit_neural_network(X_train_tensor1,y_train_tensor1,X_test_tensor1,y_test,model)
    else:
        input_size = 784
        hidden_size = 500
        num_classes = 5
        #learning_rate = 0.01
        model = NeuralNetwork(input_size, hidden_size, num_classes)

        X1, y1 = model.split(model.load_data("df_even"))
        X_train, X_test, y_train, y_test = train_test_split(X1, model.encode_even(y1), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)/255
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        neu.fit_neural_network(X_train_tensor,y_train_tensor,X_test_tensor,y_test,model)
        
        print("Before freezing:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)


        for param in model.fc1.parameters():
            param.requires_grad = False
        #Transfer Learning training    
        num_ftrs = model.fc2.in_features
        model.fc2 = nn.Linear(num_ftrs, 5)

        X2, y2 = model.split(model.load_data("df_odd"))
        X_train, X_test, y_train, y_test = train_test_split(X2, model.encode_odd(y2), test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)

        X_train_tensor1 = torch.tensor(X_train, dtype=torch.float32)/255
        y_train_tensor1 = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor1 = torch.tensor(X_test, dtype=torch.float32)/255
        y_test_tensor1 = torch.tensor(y_test, dtype=torch.float32)

        neu.fit_neural_network(X_train_tensor1,y_train_tensor1,X_test_tensor1,y_test,model)



print("\nAfter freezing:")
for name, param in model.named_parameters():
    print(name, param.requires_grad)

def main():
    if __name__=="main":
        main()