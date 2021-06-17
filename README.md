# RL-ARZ-PPO

This is the application of reinforcement learning (RL) for Aw-Rascle-Zhang (ARZ) Traffic Model using proximal poloicy optimization ([PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1)). The implementation is based on "[PyTorch Implementations of Reinforcement Learning Algorithms](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)", and modified with the application of ARZ simulator.

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [ARZ Simulator](https://github.com/saehong/gym_arz_sim)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

## How to run

* In the "settings_file.py", the parameters of ARZ simulator are defined. For RL training configuration, the hyper-paramters are defined in "main.py".

* You can simply run the code and reproduce the results by following:

```bash
# Training
python main.py

```

* After training the RL agent, one can evaluate the performance of RL controller by executing:
```bash
# Evaluation & Visualization
python evaluation_RL.py

```



## Results



### BackStepping Outlet Control
![BKST_Outlet](imgs/ARZ_BKST_Outlet.png)


### RL Outlet Control

![RL_Outlet](imgs/ARZ_RL_Outlet.png)

### P Inlet Control

![P_Inlet](imgs/ARZ_P_Inlet.png)

### RL Inlet Control

![RL_Inlet](imgs/ARZ_RL_Inlet.png)

### PI Outlet & Inlet Control

![PI_OutletInlet](imgs/ARZ_PI_Outlet_N_Inlet.png)

### RL Outlet & Inlet Control

![RL_OutletInlet](imgs/ARZ_RL_Outlet_N_Inlet.png)


## Bibtex

## Contacts

Saehong Park: sspark@berkeley.edu

Huan Yu: huy015@ucsd.edu




