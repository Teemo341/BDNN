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

* common/manual/data:  
common: the diffusion modules are initialized by default  
manual: the diffusion modules are initialized by intuition  
data: the diffusion modules are initialized by the std of drift modules  

***

* without_OOD/with_OOD:  
<font color=red>This experiment is done by </font>  
without_OOD: donot use OOD loss  
with_OOD: add OOD noise

***

* noise/cifar:  
<font color=red>This experiment is done by </font>  
noise: take noise as OOD  
cifar: take cifar-10 as OOD  

***

* initialized/notinitialized:  
initialized: use resnet to initialize drift  
notinitialized: do not initialize drift