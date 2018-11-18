#coding: UTF-8
import random
import numpy as np
import chainer
import argparse
from chainer import cuda,serializers,optimizers,Variable
import Waveform_Classfy_Net as wcn
'''
Waveform_Classfy_Net学習
'''
#引数設定
parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--train_epoch','-e',default=10, type=int,
                    help='学習回数')
parser.add_argument('--train_name' ,'-t',default="./train.txt", type=str,
                    help='学習データファイル名(default=./train.txt)')
parser.add_argument('--save' ,'-s',default="./model.npz", type=str,
                    help='モデル保存先(default=./model.npz)')
args = parser.parse_args()#引数
#gpu利用する場合
def use_gpu(model,number):
    import cupy
    cupy.random.seed(0)
    cuda.check_cuda_available()
    cuda.get_device(number).use()
    model.to_gpu()
    xp=cuda.cupy

# 引数ファイルの学習データを前処理して，データセットの生成
def preprocess(train_name):
    INPUT_ELEMENT_NUM=120#入力データ一つの要素数
    input_data=[]#入力データ
    teach=[]#データの正解のラベル
    import csv
    data_raw = open(train_name)#csvデータを文字列変換
    csv_file=csv.reader(data_raw,delimiter=",",lineterminator="\r\n")#csvのリストとして読む
    for line in csv_file:#csvの各行ごと処理
        input_data.append(line[:-1])
        teach.append(line[-1])#lineの最後の要素，正解データラベル
    input_data=xp.array(input_data, dtype=xp.float32)#chainerで扱うために変換
    teach=xp.array(teach,dtype=xp.int32)
    input_data=input_data.reshape(len(input_data),1,1,INPUT_ELEMENT_NUM)#chainerのCNNで扱うために変換
    return input_data,teach

# 学習処理を行いlossを返す
def learning(input_data,teach):
    model.cleargrads()#
    loss = model.loss(input_data,teach)#モデルの損失をとる
    loss.backward()#逆誤差伝搬
    optimizer.update()# 勾配の更新
    return loss

if __name__ == '__main__':
    xp=np#numpy
    random.seed(0)# set Python random seed
    xp.random.seed(0)# set NumPy random seed
    model = wcn.Waveform_Classfy_Net()#学習するネットワーク
    optimizer = optimizers.Adam()# 勾配降下法設定 最適化のアルゴリズムには Adam を使用
    optimizer.setup(model)# modelのパラメータをoptimizerにセット
    if(args.gpu>=0):
        use_gpu(model,args.gpu)
    frequency=1#途中経過lossを学習何回ごとに表示するか
    input_data,teach=preprocess(args.train_name)#学習データ
    # train_epoch回学習を繰り返す
    for i in range(1,args.train_epoch+1):
        loss=learning(input_data,teach)
        if i%frequency==0 or i==1:
            print( "%7d th trial loss:%17.10f" % (i, loss.data) )# 現状のMSEを表示
    #モデル保存
    model.to_cpu() # CPUで計算できるようにしておく
    serializers.save_npz(args.save, model) # npz形式で書き出し
    print("Success learning!! yeaaaaaahhhhhhh!!!!!!!!!!!!!!!!!")
