#### 自适应码率
AI工作：
| 工作 | 方法 | 源码 | 数据集 | 实验环境 | 评估方法 |
| --- | --- | --- | --- | --- | --- |
| [Fugu](https://www.usenix.org/system/files/nsdi20-paper-yan.pdf)[NSDI21] |搭建了一个大规模的播放网站，在公开的播放环境下对不同的算法进行测试 |  |  |  |
| [OnRL](https://dl.acm.org/doi/10.1145/3372224.3419186)[Mobicom20] | 针对拟合不强的缺陷，提出在线地去训练RL模型，使其能更好地拟合当前的网络环境 |  |  |  |
| [Stick](https://godka.github.io/Infocom_20_Stick.pdf)[Infocom20] | 采用基于缓存和基于强化学习的混合方法，保留了简单高效的特点和DRL灵活的优点 |  |  |  |
| [Comyco](https://arxiv.org/pdf/1908.02270)[MM19] | 利用模仿学习来改进RL | [\[code\]](https://github.com/thu-media/Comyco) |  |  |
| [Concerto](https://dl.acm.org/doi/10.1145/3300061.3345430)[Mobicom19] | 利用模仿学习来改进RL |  |  |  |
| [PiTree](https://zilimeng.com/papers/pitree-mm19.pdf)[MM19] | 针对强化学习的可解释性不强，通过决策树模型来拟合训练得到的强化学习模型，保证精度的基础上提高可解释性。 | [\[code\]](https://github.com/transys-project/pitree) |  |  |
| [Oboe](https://engineering.purdue.edu/~isl/papers/sigcomm18-final128.pdf)[sigcomm18] | 提出一个自适应的参数选择框架 |  |  |  |
| [Pensieve](https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf)[Sigcomm17] | 单纯的RL算法 | [\[code\]](https://github.com/hongzimao/pensieve) | FCC提供的宽带数据集和挪威收集的3G/HSDPA移动数据集 |  | QoE
| [Update](https://openreview.net/pdf?id=SJlCkwN8iV) [ICML19] | RL算法 |  |  |  |

