# WaveHeight-Prediction
## 基于粒子群算法优化的BPNN和ElM对海浪高度的预测
版本python3.7，安装模块Pytorch、sklearn和numpy；
wavewindF.nc是数据文件，记录2016年渤海海浪数据；
后缀.pkl是神经网络参数存储文件，在运行代码之后生成；
bpnn.py、elm.py是bp神经网络和极限学习机；
ipso.py是粒子群算法效果演示模块；
ipso-bpnn.py、ipso-elm.py是结合了优化算法的bp神经网络和极限学习机；
ncread.py、data_cache.py、swhplot.py分别是读取文件模块、数据处理模块和海浪高度绘图模块，对于不同的数据集需要修改ncread模块的代码。
