# Explain what ablation experiments will be done

* BDNN/normal:
BDNN: our model
normal: normal BNN

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
<font color=red>This experiment is done by `train.py` </font>
without_OOD: use noises as OOD
with_OOD: use cifar-10 as OOD