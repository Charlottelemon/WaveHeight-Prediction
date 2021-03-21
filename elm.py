import torch 
from torch.autograd import Variable 
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import numpy as np
from swhplot import caloss
from data_cache import *
  
torch.manual_seed(1) # 设定随机数种子 
#mm = MinMaxScaler()
#y_train = torch.tensor(mm.fit_transform(labeltrain)).float()    #归一化处理
# 神经网络结构 
net = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(5, 1)
    #torch.nn.ReLU()
    )

def save():
  ihlayer = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.Sigmoid()
  )
  hiddenout = ihlayer(xtrain).squeeze(-1)
  H = np.linalg.pinv(hiddenout.data.numpy().reshape(len(labeltrain),5))    #求广义逆
  T = labeltrain.data.numpy().reshape(len(labeltrain))    #矩阵转置
  beta = np.dot(H,T)    #矩阵相乘
  beta = torch.tensor(beta).float()
  
  net.state_dict()['0.weight'].copy_(ihlayer.state_dict()['0.weight'])
  net.state_dict()['0.bias'].copy_(ihlayer.state_dict()['0.bias'])
  net.state_dict()['2.weight'].copy_(beta)
  net.state_dict()['2.bias'].copy_(torch.tensor(0))
  # 保存神经网络 
  torch.save(net.state_dict(), 'elmnet_params.pkl') # 只保存神经网络的模型参数 
  
# 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构 
def reload_params(): 
  net.load_state_dict(torch.load('elmnet_params.pkl')) 
  loss_function = torch.nn.MSELoss()
  prediction = net(xtest).squeeze(-1)
  #prediction = mm.inverse_transform(prediction.data.numpy())
  #prediction = torch.tensor(prediction)
  caloss(prediction,labeltest)
  #测试集图像 
  plt.title('test_net') 
  param = np.polyfit(labeltest.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)
  p = np.poly1d(param,variable='x')
  rsquare = 1 - loss_function(labeltest,prediction).data.numpy()/np.var(labeltest.data.numpy())    #计算R方
  plt.scatter(labeltest.data.numpy(), prediction.data.numpy()) 
  plt.xlabel('ytest_label')
  plt.ylabel('ytest_prediction')
  plt.plot(labeltest.data.numpy(), p(labeltest.data.numpy()),'r--') 
  plt.text(max(labeltest.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
  plt.show()

  prediction = net(xtrain).squeeze(-1)
  #prediction = mm.inverse_transform(prediction.data.numpy())
  #prediction = torch.tensor(prediction)
  caloss(prediction,labeltrain)
  #训练集图像
  plt.title('train_net') 
  param = np.polyfit(labeltrain.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)
  p = np.poly1d(param,variable='x')
  rsquare = 1 - loss_function(labeltrain,prediction).data.numpy()/np.var(labeltrain.data.numpy())    #计算R方
  plt.scatter(labeltrain.data.numpy(), prediction.data.numpy()) 
  plt.xlabel('ytrain_label')
  plt.ylabel('ytrain_prediction')
  plt.plot(labeltrain.data.numpy(), p(labeltrain.data.numpy()),'r--') 
  plt.text(max(labeltrain.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
  plt.show()
 
#save() 
reload_params()