#f <= f_avg,w = wmin + (wmax - wmin(f-fmin))/(f_avg-fmin)
#f > f_avg,w=wmax
#c1 = (c1i - c1f)(Iter - iter)/Iter + c1f
#c2 = (c2i - c2f)(Iter - iter)/Iter + c2f

import numpy as np

swarmsize = 500    #种群大小
partlen = 10
wmax,wmin = 0.9,0.4
c1i = c2f = 2.5
c1f = c2i = 0.5
Iter = 200    #迭代次数

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
    randompv = (np.random.rand()-0.5)*2    #返回（-1，1）随机数
    return randompv

def initswarm():
    vswarm,pswarm = np.zeros((swarmsize,partlen)),np.zeros((swarmsize,partlen))
    for i in range(swarmsize):
        for j in range(partlen):
            vswarm[i][j] = getrange()
            pswarm[i][j] = getrange()
    return vswarm,pswarm
    
def getfitness(pswarm):
    pbest = np.zeros(partlen)
    fitness = np.zeros(swarmsize)
    for i in range(partlen):
        pbest[i] = 2.3
    
    for i in range(swarmsize):
        yloss = pswarm[i] - pbest
        for j in range(partlen):
            fitness[i] += abs(yloss[j])
    return fitness

def getpgfit(fitness,pswarm):
    pgfitness = fitness.min()
    pg = pswarm[fitness.argmin()].copy()
    return pg,pgfitness

vswarm,pswarm = initswarm()
fitness = getfitness(pswarm)
pg,pgfit = getpgfit(fitness,pswarm)
pi,pifit = pswarm.copy(),fitness.copy()

for iter in range(Iter):
    if pgfit <= 0.001:
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
    #更新局部最优粒子
    for i in range(swarmsize):
        if fitness[i] < pifit[i]:
            pifit[i] = fitness[i].copy()
            pi[i] = pswarm[i].copy()

for j in range(swarmsize):
    if pifit[j] < pgfit:
        pgfit = pifit[j].copy()
        pg = pi[j].copy()
print(pg)
print(pgfit)
print(iter)
