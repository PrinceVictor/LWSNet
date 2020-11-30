# An End-To-End Light-Weighted Muilt-Stage Stereo Matching Network(飞浆/PaddlePaddle Version)  

This repository is the code(PaddlePaddle version) of Our works **LW-SMNet** in C4-AI Match for Team Stereo Free(双目纵横)


## Contents

1. [Requirements](#Requirements)

### Requirements  

Our System version is Ubuntu20.04 with Graphics card Titan Xp.  

#### Enviroment Dependencies:  
- Anaconda or Miniconda
- Python3 or Later
- PaddlePaddle 2.0rc(GPU Version)
- OpenCV 4.0 or Later 

We provided [paddle_env.yml](paddle_env.yml) to install neccesarry dependencies directly throught conda.
```
conda env create -f paddle_env.yml
```

#### Dataset:
- KITTI2015(Finetune)
- Sceneflow(Pre-Train)


Prepare 
