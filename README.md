## AlphaGo Zero for Connect 4
### 模型训练
本项目中的模型以[论文](https://www.nature.com/articles/nature24270)中的模型结构为基础进行简化。模型首先采用3层<卷积-Batch Normalization>结构，分别包括32，64，128个3x3卷积过滤器，进行特征提取，接着分别传入策略预测网络和局势预测网络。策略预测网络包含一层卷积层（2个1x1卷积过滤器），Batch Normalization，最后传入Softmax分类器进行预测。局势预测网络包含一层卷积层（1个1x1卷积过滤器），Batch Normalization，最后利用tanh函数对局面评分。所有激活函数均为ReLU，模型通过Momentum SGD训练。

模型使用Tensorflow(1.5)训练，可以在`Model/Alpha`文件夹中找到。其中`Model/Alpha/train.py`中包含训练部分代码。`Model/Alpha/model.py`中定义网络模型。`Model/game.py`定义游戏环境相关的API。`Model/agent.py`中包括一些Agent类用于对战。`Model/self_play.py`用于收集训练数据。训练命令：
```
python -m Model.Alpha.train
```

### 人机对战
#### Command line 版
命令行中输入：
```
python play.py is_black
```
其中is_black=1为先手，0为后手。

#### UI版
游戏界面在Unity AssetStore中免费的[Connect 4 Starter Kit](https://assetstore.unity.com/packages/templates/connect-four-starter-kit-19722)基础上进行修改。利用UDP Socket进行客户端和服务器之间的通信。在进入游戏前需要运行`server.py`启动服务器，并指定相应的服务器地址和端口，默认地址是localhost:5555。
```
python server.py ip_address port_number
```

解压Connect4.zip, 双击`Connnect4/Connect4.exe`（Windows系统下）进入游戏界面。用户需要在初始化界面中输入服务器的地址和端口，选择先后手，点击Start进入游戏。游戏结束后点击PLAY AGAIN重新开始。
