#coding: UTF-8
# Numpy
import numpy as np
# Chainer
import chainer
import math
import random

from chainer import Chain
import chainer.functions as F
import chainer.links as L
'''
CNNにより波形データを2分類するネットワーク
特徴マップ:120→convolutionolution:(120-15)/2+1=106 → pooling:(106-2)/2+1=53
'''
class Waveform_Classfy_Net(Chain):
    def __init__(self):
        # インスタンス変数
        self.output_data=[]#出力層のデータ
        self.convolution_size=(1,15)#畳み込みサイズ
        self.convolution_stride=1#畳み込みのストライド
        self.pool_size=(1,2)#プーリングのサイズ
        self.pool_stride=2#プーリングのストライド
        super(Waveform_Classfy_Net, self).__init__(
            # (入力チャネル，出力チャネル(フィルタ枚数)，(フィルタサイズ縦，横),...)
            convolution=L.Convolution2D(1, 16, self.convolution_size, stride=self.convolution_stride),
            linear = L.Linear(None, 2),#2分類
        )
    # 順伝播，損失関数の計算
    def loss(self,input_data,teach):
        h =self.convolution(input_data)
        h =F.relu(h)
        h=F.max_pooling_2d(h, self.pool_size,stride=self.pool_stride, cover_all=False)
        h=self.linear(h)
        self.output_data = h
        return F.softmax_cross_entropy(self.output_data,teach)
