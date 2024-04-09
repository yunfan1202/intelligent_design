# Intellegent Design (智能设计)

## Motivation of this repository

1. 大多数人可能只想快速地在自己的笔记本上，简单的玩一玩，体验一下AI应用，看看跟自己的专业有无结合点，该文档希望能整理一些能快速运行起来的项目。
2. 面向设计等非纯计算机专业的本科、研究生的AI技术使用教程整合。该文档希望持续归纳整理使用AI技术的经验和踩过的坑。
3. 开源这样的归纳整理，希望对所有对AI感兴趣的各类跨学科专业的同学都有所帮助。
4. 开源的形式可以接受大家的审阅，得到大家的建议，从而可以持续不断地改进。

**Notice:**
1. 本文档主要是收集整理已有的工作(以及其他详细的教程)，让同学们能更容易地运行起AI项目来，以将更多的注意力放在设计工作上。如果对具体的技术细节感兴趣，最好去原GitHub和原论文里学习。
2. 本文档涉及到的AI技术都尽量使用统一的稳定环境(比如Pytorch, python 3.8)运行，踩过的坑也会记录下来，希望尽量减少应用AI的成本。如果实在是有不太能兼容的模型，那就按照原项目的要求单独再创一个虚拟环境吧。
3. 本文档里选择的AI技术并不一定是当前最强的，而是会综合考虑效果和容易运行的程度等因素。
4. 本文档将持续更新，欢迎大家提出宝贵建议。文档上次更新时间：2024年2月18日。

## 环境配置

详情见[这里](./environment.md)

由于大部分人都是使用Windows系统，所以本文档涉及到的所有AI技术均在Windows上测试并成功运行。理论上来说Linux应该是更容易的，如果您使用的是MacOS或者Linux系统，可以自行查阅相关资料。
[MacOS安装pytorch](https://zhuanlan.zhihu.com/p/168748757)
，[Linux安装pytorch](https://zhuanlan.zhihu.com/p/642347131)

**需要的python库**(库是写好的可以直接引用的代码包)：本文档尽量在同一个环境中运行所有的模型，以便技术之间的组合应用，具体用到的包在[requirements.txt](./requirements.txt)中。

**如果您只是单独玩其中某些项目的话，推荐直接按照步骤运行，然后缺什么库就```pip install 库的名字==版本```装什么库**，***版本参考[requirements.txt](./requirements.txt)***（一般来说安装最新的库也不会有什么问题，但是如果涉及到诸如transformers这样的库，不同版本可能会报错），也可以通过```pip install -r requirements.txt```一次性全部装好。如果国内安装比较慢，记得[更换pip源](https://zhuanlan.zhihu.com/p/127275233)。

举例：如果需要安装transformer这个库，运行的时候这个语句就是 pip install transformers==4.19.2，"4.19.2"这个版本号可以在requirements.txt里面查到。


## 智能感知 (Perception)
本文档整理了大部分先进、通用、运行友好的AI感知应用，目前主要包含计算机视觉相关技术：图像的分类、分割、深度估计、边缘检测，人脸检测与分类，人体姿态估计，以及文本，语音的识别与情感分类等任务。


详情见[这里](Perception/perception.md)

目前已整理：
[CLIP](https://github.com/openai/CLIP), 
[BLIP](https://github.com/salesforce/BLIP),
[Depth-Anything](https://github.com/LiheYoung/Depth-Anything)深度估计,
[Edge_detection](https://github.com/yunfan1202/Delving-into-Crispness)边缘检测,
[Face](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)人脸检测与识别,
[Pose estimation](https://github.com/Hzzone/pytorch-openpose)人体姿态估计等。

您还可以自行尝试其他AI项目：
（视觉、语音、文本等任务：）


## 智能生成 (Generation)
由于生成类技术极快的发展速度，本文档主要整理了stable diffusion和controlnet等相关技术。


详情见[这里](Generation/generation.md)

目前已整理：
[ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly), 
[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)等

您还可以自行尝试以下其他AI项目：

[InstantID](https://github.com/InstantID/InstantID)

## Citation

若该文档对您有所帮助，请在页面右上角点个Star⭐支持一下，谢谢！

如果转载该文档的内容，请注明出处：[https://github.com/yunfan1202/intellegent_design](https://github.com/yunfan1202/intellegent_design)。

## Acknowledgements
1. 本文档主要是收集并在已有的工作上整理，首先感谢所有优秀的开源技术项目的贡献！
2. 本文档内容由[本人](https://github.com/yunfan1202)与[本实验室](http://design.hnu.edu.cn/info/1023/5787.htm)的老师同学们共同完成，感谢大家的贡献！
3. 本文档受[learning_research](https://github.com/pengsida/learning_research)启发不少，这是个非常优秀的科研经验总结分享 (针对计算机视觉、图形学领域)，希望本文档能配合这份科研经验，在为减少技术门槛方面提供助力。