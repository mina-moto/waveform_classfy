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
特徴マップ:120→c1:(120-15)/2+1=106 → p:(106-2)/2+1=53 →　c2:(53-5)/2+1=25 → 2
'''
class Waveform_Classfy_Net(Chain):
    def __init__(self):
        # インスタンス変数
        self.input_data=[]#入力層
        self.output_data=[]#出力層
        self.conv_size=(1,15)#畳み込みサイズ
        self.conv_stride=1#畳み込みのストライド
        self.pool_size=(1,2)#プーリングサイズ
        self.pool_stride=2#プーリングのストライド
        self.teach_data=[]#教師データ
        super(Multi_CNN_Autoencoder, self).__init__(
            # (入力チャネル，出力チャネル(フィルタ枚数)，(フィルタサイズ縦，横),...)
            conv=L.Convolution2D(1, 16, self.conv_size, stride=self.conv_stride),
            linear = L.Linear(None, 2),#2分類
        )
    # ネットワーク接続，活性化関数で推定,損失関数の計算
    def loss(self):
        h =self.conv(self.input_data)
        h =F.relu(h)
        h=F.max_pooling_2d(h, self.pool_size,stride=self.pool_stride, cover_all=False)
        h=self.Linear(h)
        self.output_data = F.softmax_cross_entropy(h,t)
        return self.output_data
