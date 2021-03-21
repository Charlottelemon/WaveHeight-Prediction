import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_cache import *

def swhplot():
    #训练集海浪高度
    plt.title('train_swh')  
    plt.plot(swh[0:16104:1]) 
    plt.xlabel('data_no')
    plt.ylabel('Swh')
    plt.show()
    #测试集海浪高度
    plt.title('test_swh')  
    plt.plot(swh[16104:17568:1]) 
    plt.xlabel('data_no')
    plt.ylabel('Swh')
    plt.show()

def caloss(prediction,label):
    mae = torch.nn.L1Loss()(prediction,label).data
    mse = torch.nn.MSELoss()(prediction,label).data
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((prediction.data.numpy() - label.data.numpy()) / label.data.numpy())) * 100
    premean = prediction*0+torch.tensor(np.mean(prediction.data.numpy()))
    nse = 1-(mse/torch.nn.MSELoss()(premean,label).data)
    print(mae)
    print(rmse)
    print(mape)
    print(nse)

def swhfullin(xtest,ytest,flag):
    ytest = torch.unsqueeze(torch.tensor(ytest).float(), dim=1)   #升维
    xtest = torch.unsqueeze(torch.tensor(xtest).float(), dim=1)
    xtest = Variable(xtest, requires_grad=True)

    bpnet = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.ReLU(), 
    torch.nn.Linear(5, 1),
    torch.nn.Softplus()
    )
    elmnet = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(5, 1)
    )
    bpnet.load_state_dict(torch.load('net_params.pkl'))
    prediction1 = bpnet(xtest).squeeze(-1)
    #elmnet.load_state_dict(torch.load('elmnet_params.pkl'))
    #prediction2 = elmnet(xtest).squeeze(-1)
    bpnet.load_state_dict(torch.load('ipsoBP_params.pkl'))
    prediction3 = bpnet(xtest).squeeze(-1)
    elmnet.load_state_dict(torch.load('ipsoELM_params.pkl'))
    prediction4 = elmnet(xtest).squeeze(-1)
    #绘制图像
    if flag == 'hh':
        plt.title('bohai_preswh')
    else:
        plt.title('yellow_preswh')
    plt.xlabel('data_no')
    plt.ylabel('swh_pre')
    plt.plot(ytest.data.numpy(),'g',lw=1.5,label='truth')
    plt.plot(prediction1.data.numpy(),'r',lw=1.5,label='bpnn')
    #plt.plot(prediction2.data.numpy(),'g-')
    plt.plot(prediction3.data.numpy(),'b',lw=1.5,label='ipso-bp')
    plt.plot(prediction4.data.numpy(),'gray',lw=1.5,label='ipso-elm') 
    plt.legend(loc='upper left')
    plt.show()

yhbtest = []
xhbtest = []
for i in range(30):
    yhbtest.append(labeltest[i*48]);
    xhbtest.append(x_test[i*48]);     #黄海8:00数据
swhfullin(xhbtest,yhbtest,'hh')
for i in range(30):
    yhbtest[i] = labeltest[i*48+6];
    xhbtest[i] = x_test[i*48+6];     #渤海8:00数据
swhfullin(xhbtest,yhbtest,'bh')