# chainer-FCN

FCN (Fully Convolutional Network) is deep fully convolutional neural network architecture for semantic pixel-wise segmentation. This is implementation of "https://arxiv.org/abs/1605.06211" by using Chainer which is a neural networks library. FCN can train by using any size of image, but I trained this network using the images of the same size (256 * 256).

## Parameter

You can download the parameter from [here](https://drive.google.com/drive/folders/0B0bX7_mwN48ZTmJQOUZLeERESDQ).

!Caution: I trained the model only 100 epochs. If you wanna use this model practicaly, you have to do fine-tuning.

## Usage

### train

```
$ python train.py -g <gpu:id> -tr <path to train dataset> -ta <path to target dataset> -tt <image file names text> -e <epoch> -b <batchsize>
```
You can select else options, please read script directly.

### predict
```
$ pyton predict.py -g <gpu:id> -i <path to image> -w <path to weight>
```

## Example

+ person

![person](https://raw.githubusercontent.com/k3nt0w/chainer_fcn/master/out/pred_2009_005056.png)

+ horse

![horse](https://raw.githubusercontent.com/k3nt0w/chainer_fcn/master/out/pred_2010_004856.png)

+ bus
  
![person](https://raw.githubusercontent.com/k3nt0w/chainer_fcn/master/out/pred_2011_000969.png)
