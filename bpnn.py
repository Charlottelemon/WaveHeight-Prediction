import torch 
from torch.autograd import Variable 
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import numpy as np
from swhplot import caloss
from data_cache import *
  
torch.manual_seed(1) # 设定随机数种子 
#mm = MinMaxScaler()
#ytrain = torch.tensor(mm.fit_transform(labeltrain)).float()    #归一化处理
# 将待保存的神经网络定义在一个函数中 
def save(): 
  # 神经网络结构 
  net1 = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.ReLU(), 
    torch.nn.Linear(5, 1),
    torch.nn.Softplus()
    ) 
  optimizer = torch.optim.SGD(net1.parameters(), lr=0.3) 
  loss_function = torch.nn.MSELoss() 
  loss_array = []
  # 训练部分 
  for i in range(500): 
    prediction = net1(xtrain) 
    prediction = prediction.squeeze(-1)
    loss = loss_function(prediction, labeltrain) 
    loss_array.append(loss.data)
    if loss.data <= 0.04:
      break
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
  # 误差曲线 
  plt.title('train_loss')  
  plt.plot(loss_array) 
  plt.ylabel('loss_data')
  plt.xlabel('iter_num')
  plt.show()
  # 保存神经网络 
  torch.save(net1.state_dict(), 'net_params.pkl') # 只保存神经网络的模型参数 
  
# 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构 
def reload_params(): 
  # 首先搭建相同的神经网络结构 
  net2 = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.ReLU(), 
    torch.nn.Linear(5, 1), 
    torch.nn.Softplus()
    ) 
  # 载入神经网络的模型参数 
  net2.load_state_dict(torch.load('net_params.pkl')) 
  loss_function = torch.nn.MSELoss()
  prediction = net2(xtest).squeeze(-1)
  #prediction = mm.inverse_transform(prediction.data.numpy())
  #prediction = torch.tensor(prediction)
  caloss(prediction,labeltest)
  #测试集图像
  plt.title('test_net') 
  param = np.polyfit(labeltest.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)    #线性回归方程拟合
  p = np.poly1d(param,variable='x')    #输出方程式
  rsquare = 1 - loss_function(labeltest,prediction).data.numpy()/np.var(labeltest.data.numpy())    #计算R方
  plt.scatter(labeltest.data.numpy(), prediction.data.numpy()) 
  plt.xlabel('ytest_label')
  plt.ylabel('ytest_prediction')
  plt.plot(labeltest.data.numpy(), p(labeltest.data.numpy()),'r--') 
  plt.text(max(labeltest.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
  plt.show()

  prediction = net2(xtrain).squeeze(-1)
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