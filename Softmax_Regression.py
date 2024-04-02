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

#mnist = fetch_openml("mnist_784",version=1,parser='auto')
#train_images, train_labels = mnist.data,mnist.target
#print(train_images.shape,train_labels.shape)
#df = train_images
#df['Class'] = train_labels
#df.head()
#df_even = df[(df["Class"]).astype(int)%2==0]
#df_odd = df[(df["Class"]).astype(int)%2!=0]
#df_even.to_csv("df_even.csv",index=False)
#df_odd.to_csv("df_odd.csv",index=False)
class Softmax(nn.Module):
  def __init__(self, n_input_features,n_out):

    super(Softmax, self).__init__()
    self.linear = nn.Linear(n_input_features,n_out)

  def softmax(self,z):
    return torch.exp(z)/torch.sum(torch.exp(z),axis = 1,keepdims= True)

  def forward(self, x):
    z = self.linear(x)
    y_pred = self.softmax(z)
    return y_pred

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

  def decode_odd(self,y_onehot):
    Y_pred = y_onehot@ np.array([1,3,5,7,9]).reshape(-1,1)
    return Y_pred

  def encode_even(self,y):
    Y = [(np.array([0,2,4,6,8]) == i).astype(int) for i in y ]
    Y = np.array(Y)
    return Y

  def decode_even(self,y_onehot):
    Y_pred = y_onehot@ np.array([0,2,4,6,8]).reshape(-1,1)
    return Y_pred
  def plot(self, losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Losses VS Epochss Graph for Softmax Regression")
    plt.show()
  def confusiom_matrix_odd(self,y,pred):
    
    ax = plt.subplot()
    cm = confusion_matrix(y,pred)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=[1, 3, 5, 7, 9], yticklabels=[1, 3, 5, 7, 9]);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');

  def confusiom_matrix_even(self,y,pred):
    
    ax = plt.subplot()
    cm = confusion_matrix(y,pred)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=[0, 2, 4, 6, 8], yticklabels=[0, 2, 4, 6, 8]);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
      #ax.xaxis.set_ticklabels(T5_lables); ax.yaxis.set_ticklabels(T5_lables);


  def fit_foftmax(self,X_train_tensor,y_train_tensor,X_test_tensor,y_test,model,num_epochs,learning_rate):
    start = time.time()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    losses = []
    for epoch in range(num_epochs):

        # Forward pass and loss
        y_pred = model.forward(X_train_tensor)
        loss = criterion(y_train_tensor,y_pred)
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
        y_predicted_cls =(y_predicted >=np.max(y_predicted, axis=1).reshape(-1,1)).astype(int) #use threshold=0.5 to define the classes

        user = input("Choose 1 for ODD and 2 for EVEN :")
        if user == "2":
          y_predicted_cls = self.decode_even(y_predicted_cls)
          y_test = self.decode_even(y_test)
          accuracy = np.mean(y_predicted_cls == y_test)
          print(f'accuracy: {accuracy.item():.4f}')
          self.plot(losses)
        elif user == "1":
          y_predicted_cls_odd = self.decode_odd(y_predicted_cls)
          y_test_odd = self.decode_odd(y_test)
          accuracy = np.mean(y_predicted_cls == y_test)
          print(f'accuracy: {accuracy.item():.4f}')
          self.plot(losses)
        else:
          print("Sorry choose the correct number")
    end = time.time()
    print(f"Running time = {(end -start):.3f}")
    if user== "2":
      self.confusiom_matrix_even(y_test,y_predicted_cls)
    else:
      self.confusiom_matrix_odd(y_test_odd,y_predicted_cls_odd)
      print(np.unique(y_test_odd),np.unique(y_predicted_cls_odd))

model_s = Softmax(784,5)
X, y = model_s.split(model_s.load_data("df_even"))
X_train, X_test, y_train, y_test = train_test_split(X, model_s.encode_even(y), test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)/255
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)/255
model_s.fit_foftmax(X_train_tensor,y_train_tensor,X_test_tensor,y_test,model_s,300,0.01)