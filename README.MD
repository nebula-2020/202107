# 设计稿

## 问题分析

### 应该统计哪些量

* ~~飞机高度占总高度的比例~~
* ~~飞机宽度占总宽度的比例~~
* 飞机中点占总高度的比例
* 飞机下降速度占总高度的比例
* ~~飞机弹跳力占总高度的比例~~
* ~~飞机下降加速度占总高度的比例~~
* ~~飞机前向速度占总宽度的比例~~
* 门中点占总高度的比例
* ~~门高度占总高度的比例~~
* ~~门宽度占总宽度的比例~~
* 飞机左侧与门右侧水平距离占总宽度的比例

最后我们取：

* 飞机左侧与门右侧水平距离占当前两门间距的比例
* 飞机中点与门中点垂直距离占总高度的比例
* 飞机速度与“最大速度”的比例
* 飞机绝对高度与屏幕高度的比例

### 问题分析

* 时间不连续，最小单位为“帧”
* 状态status是连续的浮点数值
* 动作action只有2种，即“跳”和“不跳”
* 动作的结果概率为1

总结下来状态空间连续而动作空间不连续。

## 数据结构设计

### 盒子模型

全部以像素为单位。

|属性|类型|
|:-:|:-:|
|尺寸|`list`|
|速度|`list`|
|加速度|`list`|

### 游戏对象

|属性|类型|
|:-:|:-:|
|盒子|盒子模型|
|存活状态|`bool`|
|材质贴图序列|`list`|
|材质帧序号|`int`|

### 游戏

|属性|类型|
|:-:|:-:|
|玩家|游戏对象|
|门序列|`list`|
|重力|`int`|
|弹跳力|`int`|
|游戏屏幕尺寸|`tuple`|
|两门间隔|`int`|
|游戏积分|`int`|

## 算法设计

Dueling DQN.

神经元输入$\boldsymbol{s}$计算得到$\hat{a}$，然后计算$V(\boldsymbol{s})$进而得到$a^*$，然后用$a^*$当真实类别用于更新神经元权值。

问题是非线性分类，需要神经网络，输入层4个神经元，输出层2个神经元。只要玩家存活就得到正奖励。玩家通过地一个门的几率较小，将采取少量人类玩家数据初始化经验池。

## 程序设计

框架采用pygame和pytorch常用的框架。

1. 随机初始化模型；
2. 将决策网络的参数更新至目标网络；
3. 基于当前模型，计算数十种状态下每一种动作的$\bm s$、$a$、$r$、$\bm s^\prime$，存入经验池；
4. 若批量总回报充分大则认为收敛，结束；
5. 从经验池取出一批数据拟合；
6. 重复上述操作直至收敛。