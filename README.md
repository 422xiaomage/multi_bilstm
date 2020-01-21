# multi_bilstm



这是一个基于tensorflow写的2*2的双向lstm结构，用于情感分类问题，在同样的数据下效果优于textcnn，可能是因为我的数据集比较长的原因。

后期将关于双向lstm结构做完整的解释，以及优化器，反向传播和梯度消失梯度爆炸等方面的问题。这两天都将完整的写在我的csdn博客上

基于搜狐2019算法赛数据的细粒度情感分析 几乎没有对数据做什么处理，只是简单的把有实体的句子拿出来，但是发现这样做破坏了一些有用的句子，这样肯定也是没有办法解决一个句子中有多个实体情感判断的问题，这里只是一个小尝试，重在自己尝试搭模型。当然在采取一些小措施之后这单一模型也是可以到0.6的效果的

data文件夹存放的是数据以及停用词表

model文件夹存放训练好的模型文件

word2vec文件夹下是训练好的词向量模型，用的是这次比赛的训练集和测试集共同训练。

data_process.py用于数据预处理

textcnn_model.py是textcnn的模型代码

train.py是训练代码

下载下来直接运行train.py文件，就可以训练模型

如果需要改动以适应于自己的数据，主要需要修改train.py里面的参数设置部分

此次代码中还有很多需要完善的地方，将在今后的学习中继续完善


关于模型更详细的介绍可以查看本人的csdn博客：请稍等，马上更新。

word2vec模型放在了我的百度网盘里，拷贝下来放在word2vec文件夹下即可。百度云链接：链接：https://pan.baidu.com/s/1X5FYj7fubF8KZVhLZdKtwg 
提取码：q4k9 

本人初学者一枚，所以可能有写错的地方还望指出

模型太大上传不上去，还请见谅。
66/42/105/230;815;mq19950815
