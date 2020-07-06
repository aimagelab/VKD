# Robust Re-Identification by Multiple Views Knowledge Distillation

## Installation Notes

Tested with Python3.6.8 on Ubuntu (17.04, 18.04).

- Setup an empty pip environment 
- Install packages using ``pip install -r requirements.txt``
- Install torch1.3.1 using ``pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
``
- Place datasets in ``.datasets/`` (Please note you may need do request some of them to their respective authors)
- Run scripts from ```commands.txt```

Please note that if you're running the code from Pycharm (or another IDE) you may need to manually set the working path to ``PROJECT_PATH``

## MARS Example
- Create the folder ``./datasets/mars``
- Download the dataset from [here](https://drive.google.com/drive/u/1/folders/0B6tjyrV1YrHeMVV2UFFXQld6X1E)
- Unzip data and place the two folders inside the MARS folder
- Download metadata from [here](https://github.com/liangzheng06/MARS-evaluation/tree/master/info)
- Place them in a folder named ``info`` under the same path
- You should end up with the following structure:

```
PROJECT_PATH/datasets/mars/
|-- bbox_train/
|-- bbox_test/
|-- info/
```

To train ResNet-50 on MARS (teacher, first step) run:

``python ./tools/train_v2v.py mars --backbone resnet50 --num_train_images 8 --p 8 --k 4 --exp_name base_mars_resnet50 --first_milestone 100 --step_milestone 100`` 

To train a ResVKD-50 (student) for the aforementioned configuration run:

``python ./tools/train_distill.py mars ./logs/base_mars_resnet50 --exp_name distill_mars_resnet50 --p 12 --k 4 --step_milestone 150 --num_epochs 500``

You can evaluate both networks using the ``eval.py`` script:

``mars ./logs/base_mars_resnet50``

``mars ./logs/distill_mars_resnet50 --trinet_chk_name chk_di_1`` 