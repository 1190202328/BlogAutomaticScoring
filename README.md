# BlogAutomaticScoring(个人博客自动打分系统)


## 1⃣️介绍

#### 主要功能：
1.给定一篇博客，程序给定出这篇博客的原创性等级。


#### 目前包含如下[模型]：  

    -- 普通全连接模型
    -- 


## 2⃣️软件架构

* BlogAutomaticScoring(项目名)  
    * src(代码)  
    * text(文档)
   
   
## 3⃣️安装教程

1. 克隆代码
2. 前往[EDU模型下载地址]，下载pretrained_edu.pkl，pretrained_rlat.pkl，pretrained_trans.pkl，
并将这三个文件放入[saved_model文件夹]


## 4⃣️使用说明

1.  运行命令行`pip install -r requirement.txt`，安装依赖库
2.  运行[...]()


## 5⃣️参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

## 6⃣️鸣谢
1. [EDU(语篇基本单元划分模型)](https://github.com/jeffrey9977/Chinese-Discourse-Parser-ACL2020)  
2. 


[模型]: src/machine_learning/model_generator.py
[EDU模型下载地址]: https://github.com/jeffrey9977/Chinese-Discourse-Parser-ACL2020/releases
[saved_model文件夹]: src/saved_model/