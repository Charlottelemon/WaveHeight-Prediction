import torch 
#from torch.autograd import Variable 
import matplotlib.pyplot as plt 
import numpy as np
from swhplot import caloss
from data_cache import *

net = torch.nn.Sequential( 
    torch.nn.Linear(2, 5), 
    torch.nn.ReLU(), 
    torch.nn.Linear(5, 1), 
    torch.nn.Softplus()
    ) 
#粒子群算法搜索过程
swarmsize = 100
partlen = 21    #神经网络参数个数
wmax,wmin = 0.9,0.4
c1i = c2f = 2.5
c1f = c2i = 0.5
Iter = 300

def getwgh(fitness,i):
    sum = 0
    for j in fitness:
        sum += j
    if fitness[i] <= sum/swarmsize:
        w = wmin + (wmax - wmin*(fitness[i] - fitness.min()))/(sum/swarmsize - fitness.min())
    else:
        w = wmax
    return w

def getc1c2(iter):
    c1 = (c1i - c1f)*(Iter - iter)/Iter + c1f
    c2 = (c2i - c2f)*(Iter - iter)/Iter + c2f
    return c1,c2

def getrange():
    randompv = (np.random.rand()-0.5)*4
    return randompv

def initswarm():
    vswarm,pswarm = np.zeros((swarmsize,partlen)),np.zeros((swarmsize,partlen))
    for i in range(swarmsize):
        for j in range(partlen):
            vswarm[i][j] = getrange()
            pswarm[i][j] = getrange()
    return vswarm,pswarm
    
def getfitness(pswarm):
    fitness = np.zeros(swarmsize)
    loss_function = torch.nn.MSELoss()
    for i in range(swarmsize):
        params = pswarm[i]
        net.state_dict()['0.weight'].copy_(torch.tensor(np.array(params[0:10:1]).reshape(5,2)))
        net.state_dict()['0.bias'].copy_(torch.tensor(params[10:15:1]))
        net.state_dict()['2.weight'].copy_(torch.tensor(params[15:20:1]))
        net.state_dict()['2.bias'].copy_(torch.tensor(params[20]))
        prediction = net(xtrain) 
        prediction = prediction.squeeze(-1)
        fitness[i] = loss_function(prediction, labeltrain)
    return fitness

def getpgfit(fitness,pswarm):
    pgfitness = fitness.min()
    pg = pswarm[fitness.argmin()].copy()
    return pg,pgfitness

def optimi():
    vswarm,pswarm = initswarm()
    fitness = getfitness(pswarm)
    pg,pgfit = getpgfit(fitness,pswarm)
    pi,pifit = pswarm.copy(),fitness.copy()
    pgfitlist = []      #存放迭代过程中的全局最优粒子适应值
    for iter in range(Iter):
        if pgfit <= 0.01:
            break
        #更新速度和位置
        for i in range(swarmsize):
            weight = getwgh(fitness,i)
            c1,c2 = getc1c2(iter)
            for j in range(partlen):
                vswarm[i][j] = weight*vswarm[i][j] + c1*np.random.rand()*(pi[i][j]-pswarm[i][j]) + c2*np.random.rand()*(pg[j]-pswarm[i][j])
                pswarm[i][j] = pswarm[i][j] + vswarm[i][j]
        #更新适应值
        fitness = getfitness(pswarm)
        #更新全局最优粒子
        pg,pgfit = getpgfit(fitness,pswarm)
        pgfitlist.append(pgfit)
        #更新局部最优粒子
        for i in range(swarmsize):
            if fitness[i] < pifit[i]:
                pifit[i] = fitness[i].copy()
                pi[i] = pswarm[i].copy()
    #绘制粒子搜索过程全局最优的适应值变化
    plt.title('swarm_fit')  
    plt.plot(pgfitlist) 
    plt.ylabel('pg_fitness')
    plt.xlabel('iter_num')
    plt.show()
    #最后对全局最优粒子适应值和局部最优进行比较
    for j in range(swarmsize):   
        if pifit[j] < pgfit:
            pgfit = pifit[j].copy()
            pg = pi[j].copy()
    #优化完成,初始化参数
    net.state_dict()['0.weight'].copy_(torch.tensor(np.array(pg[0:10:1]).reshape(5,2)))
    net.state_dict()['0.bias'].copy_(torch.tensor(pg[10:15:1]))
    net.state_dict()['2.weight'].copy_(torch.tensor(pg[15:20:1]))
    net.state_dict()['2.bias'].copy_(torch.tensor(pg[20]))
    torch.save(net.state_dict(), 'ipsoBP_params.pkl')

# 训练部分 
def save_train():
    net.load_state_dict(torch.load('ipsoBP_params.pkl'))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05) 
    loss_function = torch.nn.MSELoss() 
    loss_array = []
    for i in range(300): 
        prediction = net(xtrain) 
        prediction = prediction.squeeze(-1)
        loss = loss_function(prediction, labeltrain) 
        loss_array.append(loss.data)
        if loss.data <= 0.01:
            break
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
    #训练集图像
    caloss(prediction,labeltrain)
    plt.title('train_net') 
    param = np.polyfit(labeltrain.squeeze(-1).data.numpy(), prediction.squeeze(-1).data.numpy(),1)
    p = np.poly1d(param,variable='x')
    rsquare = 1 - loss.data.numpy()/np.var(labeltrain.data.numpy())    #计算R方
    plt.scatter(labeltrain.data.numpy(), prediction.data.numpy()) 
    plt.xlabel('ytrain_label')
    plt.ylabel('ytrain_prediction')
    plt.plot(labeltrain.data.numpy(), p(labeltrain.data.numpy()),'r--') 
    plt.text(max(labeltrain.data),max(prediction.data),'y='+str(p).strip()+'\nRsquare='+str(round(rsquare,4)),verticalalignment="top",horizontalalignment="right")
    plt.show()
    # 误差曲线 
    plt.title('train_loss')  
    plt.plot(loss_array) 
    plt.ylabel('loss_data')
    plt.xlabel('iter_num')
    plt.show()
    # 保存神经网络 
    torch.save(net.state_dict(), 'ipsoBP_params.pkl') # 只保存神经网络的模型参数

def reload_test():
    # 载入神经网络的模型参数 
    net.load_state_dict(torch.load('ipsoBP_params.pkl')) 
    loss_function = torch.nn.MSELoss()
    prediction = net(xtest)
    prediction = prediction.squeeze(-1)
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
#运行测试
#optimi()
#save_train()
reload_test()
