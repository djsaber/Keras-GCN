# Keras-GCN
基于Keras搭建一个简单的GCN，用cora数据集对GCN进行训练，完成节点分类测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：将数据集文件解压至此<br />
2. /save_models：保存训练好的模型权重文件，包括生成器权重和判别器权重两个文件<br />

GCN概述<br />
图神经网络(Graph Neural Network, GNN)是指神经网络在图上应用的模型的统称，根据采用的技术不同和分类方法的不同，
又可以分为下图中的不同种类，例如从传播的方式来看，图神经网络可以分为图卷积神经网络（GCN），图注意力网络（GAT，缩写为了跟GAN区分），Graph LSTM等等<br />
图卷积神经网络(Graph Convolutional Network, GCN)正如上面被分类的一样，是一类采用图卷积的神经网络，
发展到现在已经有基于最简单的图卷积改进的无数版本，在图网络领域的地位正如同卷积操作在图像处理里的地位。<br /><br />


数据集：<br />
cora：包含2708篇科学出版物网络，共有5429条边，总共7种类别。<br />
数据集中的每个出版物都由一个 0/1 值的词向量描述，表示字典中相应词的缺失/存在。 该词典由 1433 个独特的词组成。<br />
链接：https://pan.baidu.com/s/1iBNmixiuORjpHjDSFephcg?pwd=52dl 提取码：52dl<br /><br />
