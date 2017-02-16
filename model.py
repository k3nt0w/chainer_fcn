import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable

class FCN(chainer.Chain):
    def __init__(self, n_class=21):
        super(FCN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            pool3=L.Convolution2D(256, n_class, 1, stride=1, pad=0),
            pool4=L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            pool5=L.Convolution2D(512, n_class, 1, stride=1, pad=0),

            upsample4=L.Deconvolution2D(n_class, n_class, ksize= 4, stride=2, pad=1),
            upsample5=L.Deconvolution2D(n_class, n_class, ksize= 8, stride=4, pad=2),
            upsample =L.Deconvolution2D(n_class, n_class, ksize=16, stride=8, pad=4),
        )
        self.train = False

    def calc(self, x, test=False):
        h = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        p3 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(p3))))))
        p4 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(p4))))))
        p5 = F.max_pooling_2d(h, 2, stride=2)

        p3 = self.pool3(p3)
        p4 = self.upsample4(self.pool4(p4))
        p5 = self.upsample5(self.pool5(p5))

        h = p3 + p4 + p5
        o = self.upsample(h)

        return o

    def __call__(self, x, t=None, train=False, test=False):
        h = self.calc(x, test)

        if train:
            loss = F.softmax_cross_entropy(h, t)
            return loss
        else:
            pred = F.softmax(h)
            return pred
