# Explain what ablation experiments will be done

* BDNN/BNN:  
BDNN: our model  
BNN: normal BNN  

***

* single/multi:  
single: the diffusion module only contains one conv layer  
linear: the diffusion module is the same as drift module  

***

* residual/linear:  
residual: the diffusion modules are isolated  
linear: the diffusion modules are connected  

***

* without_OOD/with_OOD:  
<font color=red>This experiment is done by </font>  
without_OOD: do not use OOD loss  
with_OOD: use OOD loss

***

* true/pseudo:  
true: use true label and reverse loss  
pseudo: use pseudo label and donot reverse loss  

***

* noise/cifar:  
<font color=red>This experiment is done by </font>  
noise: take noise as OOD  
cifar: take cifar-10 as OOD  

***

* common/manual/data:  
common: the diffusion modules are initialized by default  
manual: the diffusion modules are initialized by intuition  
data: the diffusion modules are initialized by the std of drift modules  

***

* initialized/notinitialized:  
initialized: use resnet to initialize drift  
notinitialized: do not initialize drift  

***
### 论文最终模型应该是 sin_res_with_true_noi_data_ini  

### 我估计 sin_res_with_true_cifar_data_ini 效果会最好  

### 要做的实验：

1. 对偶结构是否有用 BDNN_sin_res_with_true_noi_data_ini / BNN_none_none_with_true_noi_man_none  

2. OODloss会不会影响结果  
3. 用真标签还是伪标签  
4. OOD用真实数据还是噪声  
5. 单层是否会影响结果  
6. res结构还是linear结构  
7. 三种贝叶斯初始化有没有不同  
8. 从头训练会不会有提高  