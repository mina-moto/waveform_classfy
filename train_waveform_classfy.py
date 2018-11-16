#coding: UTF-8
import random
# Numpy
import numpy as np
# Chainer
import chainer
import argparse
from chainer import cuda,serializers
import Wave as MM
'''
Waveform_Classfy_Net学習
'''
#引数設定
parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--train_epoch','-e',default=1000, type=int,
                    help='学習回数(default=1000)')
parser.add_argument('--cnn' ,'-c',default=0, type=int,
                    help='畳み込みを用いるかどうか，0以上なら行う(default=0)')
parser.add_argument('--input_name' ,'-i',default="./elect_data/sample.txt", type=str,
                    help='入力データファイル名(default=./elect_data/sample.txt)')
parser.add_argument('--save' ,'-s',default="model.npz", type=str,
                    help='モデル保存先(default=model.npz)')
args = parser.parse_args()#引数

random.seed(0)# set Python random seed
np.random.seed(0)# set NumPy random seed

model = MCAE.Multi_CNN_Autoencoder()#学習するネットワーク

#gpu利用
def use_gpu():
    if args.gpu >= 0:
        # set Chainer(CuPy)
        import cupy#確かgpuないと使えない
        cupy.random.seed(0)
        cuda.check_cuda_available()
        cuda.get_device(0).use()
        model.to_gpu()
        xp=cuda.cupy

# 入力データを前処理して，モデルの入力データとする
def preprocess(self,input_name):
    INPUT_ELEMENT_NUM=120#入力データ要素数，1~121番目のデータ,0番目は日付のデータ
    data_raw = open(input_name)  # csvデータ
    for line in data_raw:# 標本データの生成
        # days.append(line.split(",")[0])
        input = xp.array([xp.float32(float(i)) for i in line.split(",")[1:INPUT_ELEMENT_NUM+1]])#1~121番目のデータ
        input = xp.array(input, dtype=xp.float32).reshape(1,1,INPUT_ELEMENT_NUM)# CNN用へ変換
        model.input_data.append(input)
    model.input_data = Variable(xp.array(model.input_data, dtype=xp.float32))# chainerの変数として再度宣言


optimizer = optimizers.Adam()# 勾配降下法設定 最適化のアルゴリズムには Adam を使用
optimizer.setup(model)# modelのパラメータをoptimizerにセット

# 学習処理
def learning(self):
    model.cleargrads()#
    loss = model.loss()#モデルの損失
    loss.backward()#逆誤差伝搬
    optimizer.update()# 勾配の更新

frequency=100#途中経過lossを学習何回ごとに表示するか
# パラメータの学習を繰り返す
for i in range(1,args.train_epoch+1):
    learning()
    if i%frequency==0 or i==1:
        print( "%7d th trial loss:%17.10f" % (i, loss.data) )# 現状のMSEを表示

#モデル保存
model.to_cpu() # CPUで計算できるようにしておく
# cPickle.dump(model, open(args.save, "wb"), -1)
serializers.save_npz(args.save, model) # npz形式で書き出し

print("Success learning!! yeaaaaaahhhhhhh!!!!!!!!!!!!!!!!!")
