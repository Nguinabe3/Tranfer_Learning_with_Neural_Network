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
#%matplotlib inline

class NeuralNetwork(nn.Module):

  def __init__(self, input_size, hidden_size, num_classes):
      super(NeuralNetwork, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return torch.softmax(x,dim=1)

  def load_data(self,df_name):
    df = pd.read_csv(df_name+".csv")
    return df

  def split(self,df):
    X = df.iloc[:,:-1].to_numpy()
    y = df['Class'].to_numpy()
    return X, y

  def encode_odd(self,y):
    Y = [(np.array([1,3,5,7,9]) == i).astype(int) for i in y ]
    Y = np.array(Y)
    return Y

  def encode_even(self,y):
    Y = [(np.array([0,2,4,6,8]) == i).astype(int) for i in y ]
    Y = np.array(Y)
    return Y

  def shuffle_data(self,X, y):
    N, _ = X.shape
    shuffled_idx = np.random.permutation(N)
    return X[shuffled_idx], y[shuffled_idx]

  def decode_odd(self,y_onehot):
    Y_pred = y_onehot@ np.array([1,3,5,7,9]).reshape(-1,1)
    return Y_pred

  def decode_even(self,y_onehot):
    Y_pred = y_onehot@ np.array([0,2,4,6,8]).reshape(-1,1)
    return Y_pred

def decode_odd(y_onehot):
  Y_pred = y_onehot@ np.array([1,3,5,7,9]).reshape(-1,1)
  return Y_pred

def decode_even(y_onehot):
  Y_pred = y_onehot@ np.array([0,2,4,6,8]).reshape(-1,1)
  return Y_pred
def plot(losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Losses VS Epochs Graph for NN")
    plt.show()
def confusiom_matrix_odd(y,pred):
  ax = plt.subplot()
  cm = confusion_matrix(y,pred)
  sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=[1, 3, 5, 7, 9], yticklabels=[1, 3, 5, 7, 9]);
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
  ax.set_title('Confusion Matrix');

def confusiom_matrix_even(y,pred):

  ax = plt.subplot()
  cm = confusion_matrix(y,pred)
  sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=[0, 2, 4, 6, 8], yticklabels=[0, 2, 4, 6, 8]);
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
  ax.set_title('Confusion Matrix');


def fit_neural_network(X_train_tensor, y_train_tensor,X_test_tensor,y_test,model):

  start = time.time()

 
  num_epochs = 300
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)


  losses = []
  
  for epoch in range(num_epochs):

    y_pred = model.forward(X_train_tensor)
    loss = criterion(y_pred,y_train_tensor)
    losses.append(loss)
    # Backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()


    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

  with torch.no_grad():
    y_predicted = model.forward(X_test_tensor).numpy()
    y_predicted_cls =(y_predicted >=np.max(y_predicted,axis=1).reshape(-1,1)).astype(int) 
    user = input("Choose 1 for ODD and 2 for EVEN :")
    if user == "2":
      y_predicted_cls = decode_even(y_predicted_cls)
      y_test = decode_even(y_test)
      accuracy = np.mean(y_predicted_cls == y_test)
      print(f'accuracy: {accuracy:.4f}')
      plot(losses)
    elif user == "1":
      y_predicted_cls_odd = decode_odd(y_predicted_cls)
      y_test_odd = decode_odd(y_test)
      accuracy = np.mean(y_predicted_cls_odd == y_test_odd)
      print(f'accuracy: {accuracy:.4f}')
      plot(losses)
    else:
      print("Sorry choose the correct number")
  end = time.time()
  print(f"Time running : {end - start}")
  if user== "2":
    confusiom_matrix_even(y_test,y_predicted_cls)
  else:
    confusiom_matrix_odd(y_test_odd, y_predicted_cls_odd)
