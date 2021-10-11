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
3. 前往[proxy_pool(代理池)]，下载代理池项目
4. 前往[BERT预训练模型]，下载*chinese_L-12_H-768_A-12*模型


## 4⃣️使用说明

1. 运行命令行`pip install -r requirement.txt`，安装依赖库
2. 运行命令行`conda activate test`，进入虚拟环境，准备运行BERT-service  
3. 运行命令行`bert-serving-start -model_dir /Users/jiangjingjing/Desktop/BERT/chinese_L-12_H-768_A-12 -num_worker=1`，启动BERT-service  
4. 运行命令行`/Users/jiangjingjing/Desktop/redis-stable/src/redis-server`，启动redis-server  
5. 运行命令行`/Users/jiangjingjing/Desktop/redis-stable/src/redis-cli`，启动redis-client  
6. 运行命令行`/usr/local/bin/python3.9 /Users/jiangjingjing/Desktop/proxy_pool-2.3.0/proxyPool.py schedule`，启动代理池爬取程序
7. 运行命令行`/usr/local/bin/python3.9 /Users/jiangjingjing/Desktop/proxy_pool-2.3.0/proxyPool.py server`，启动代理池网页客户端  


## 5⃣️参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

## 6⃣️鸣谢
1. [EDU(语篇基本单元划分模型)]
2. [proxy_pool(代理池)]
3. [PMD(分析代码相似度的软件)]
4. [BERT预训练模型]


[模型]: src/machine_learning/model_generator.py
[EDU模型下载地址]: https://github.com/jeffrey9977/Chinese-Discourse-Parser-ACL2020/releases
[saved_model文件夹]: src/saved_model/
[EDU(语篇基本单元划分模型)]: https://github.com/jeffrey9977/Chinese-Discourse-Parser-ACL2020
[proxy_pool(代理池)]: https://github.com/jhao104/proxy_pool
[PMD(分析代码相似度的软件)]: https://pmd.github.io
[BERT预训练模型]: https://github.com/google-research/bert