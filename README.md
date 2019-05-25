# CycleGAN using PyTorch
This code is inspired by the implementation by [arnab39](https://github.com/arnab39) which can be found [here](https://github.com/arnab39/cycleGAN-PyTorch).

## Requirements
- The code has been written in Python (3.5.2) and PyTorch (0.4.1)

## How to run
* To download datasets (eg. horse2zebra)
```
$ sh ./download_dataset.sh horse2zebra
```
* To run training
```
$ python main.py --training True
```
* To run testing
```
$ python main.py --testing True
```
* Try tweaking the arguments to get best performance according to the dataset.

## Results

* To view results rename the preferred checkpoint to latest.ckpt and run a test.