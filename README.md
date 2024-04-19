# CapsNet-Tensorflow

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg?style=plastic)](https://gitter.im/CapsNet-Tensorflow/Lobby)

A Tensorflow implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)



## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3
- tqdm (for displaying training progress info)
- scipy (for saving images)

## Usage
**Step 1.** Download this repository with ``git`` or click the [download ZIP](https://github.com/naturomics/CapsNet-Tensorflow/archive/master.zip) button.

```
$ git clone https://github.com/naturomics/CapsNet-Tensorflow.git
$ cd CapsNet-Tensorflow
```

**Step 2.** Download [Datasets](https://github.com/ruanxiaoli/membrane-protein-types/tree/main/Datasets)



**Step 3.** Start the training(Using the Datasets by default):

```
$ python ICNN.py

$ # If you need to monitor the training process, open tensorboard with this command
$ tensorboard --logdir=logdir
$ # or use `tail` command on linux system
$ tail -f results/val_acc.csv
```

**Step 4.** Calculate test accuracy

```
$ python main.py --is_training=False
$ # for fashion-mnist dataset
$ python main.py --dataset fashion-mnist --is_training=False
```

> **Note:** The default parameters of batch size is 128, and epoch 50. You may need to modify the ``config.py`` file or use command line parameters to suit your case, e.g. set batch size to 64 and do once test summary every 200 steps: ``python main.py  --test_sum_freq=200 --batch_size=48``

## Results
The pictures here are plotted by tensorboard and my tool `plot_acc.R`

- training loss

![total_loss](results/total_loss.png)
![margin_loss](results/margin_loss.png)
![reconstruction_loss](results/reconstruction_loss.png)




> My simple comments for capsule
> 1. A new version neural unit(vector in vector out, not scalar in scalar out)
> 2. The routing algorithm is similar to attention mechanism
> 3. Anyway, a great potential work, a lot to be built upon


