import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import sys
import os
import argparse

from model import FCN
from preprocess import load_data

parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--train_dataset', '-tr', default='dataset', type=str)
parser.add_argument('--target_dataset', '-ta', default='dataset', type=str)
parser.add_argument('--train_txt', '-tt', default='train_txt', type=str)
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--classes', default=21, type=int)
args = parser.parse_args()

n_epoch = args.epoch
n_class = args.classes
batchsize = args.batchsize
image_size = args.image_size
train_dataset = args.train_dataset
target_dataset = args.target_dataset
train_txt = args.train_txt

with open(train_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
n_data = len(names)
n_iter = n_data // batchsize
gpu_flag = True if args.gpu > 0 else False

model = FCN(n_class, gpu_flag=gpu_flag)

if args.initmodel:
    serializers.load_npz(args.initmodel, model)
    print("Load initial weight")

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer parameters.
optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_fcn')

print("## INFORMATION ##")
print("Num Data: {}, Batchsize: {}, Iteration {}".format(n_data, batchsize, n_iter))

print("-"*40)
for epoch in range(1, n_epoch+1):
    print('epoch', epoch)
    for i in range(n_iter):

        model.zerograds()
        indices = range(i * batchsize, (i+1) * batchsize)

        x = xp.zeros((batchsize, 3, image_size, image_size), dtype=np.float32)
        y = xp.zeros((batchsize, image_size, image_size), dtype=np.int32)
        for j in range(batchsize):
            name = names[i*batchsize + j]
            xpath = train_dataset+name+".jpg"
            ypath = target_dataset+name+".png"
            x[j] = load_data(xpath, crop=True, size=256, mode="data", xp=xp)
            y[j] = load_data(ypath, crop=True, size=256, mode="label", xp=xp)

        x = Variable(x)
        y = Variable(y)
        loss = model(x, y, train=True)

        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(i+1, n_iter, loss.data))
        sys.stdout.flush()

        loss.backward()
        optimizer.update()
    print("\n"+"-"*40)

if not os.path.exists("weight"):
    os.mkdir("weight")
serializers.save_npz('weight/chainer_fcn.weight', model)
serializers.save_npz('weight/chainer_fcn.state', optimizer)
print('save weight')
