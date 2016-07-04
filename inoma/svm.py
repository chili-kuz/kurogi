#!/usr/bin/env python
# coding:utf-8
"""
http://next.rikunabi.com/tech_souken/entry/ct_s03600p002315
で拾ったデータセットに対しSVMを適用

データセットの.txtファイルはソースコードと同じディレクトリ上を想定
"""

import os
from pyspark import SparkContext
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint

#plt.style.use('ggplot')

P = os.path.abspath(os.path.dirname(__file__))# スクリプトのあるディレクトリへの絶対パス取得
#訓練データ読み込み
data_t = np.loadtxt('{}/CodeIQ_auth.txt'.format(P),delimiter=' ')
#テストデータ読み込み
test_t = np.loadtxt('{}/CodeIQ_mycoins.txt'.format(P),delimiter=' ')
sc = SparkContext()


# Sparkが読み取れるようにデータを変換
data = sc.parallelize(data_t)
test = sc.parallelize(test_t)


#0番目にラベルデータを、それ以降特徴ベクトル
def parsePoint(vec):
    return LabeledPoint(vec[-1], vec[0:-1])

parsedData = data.map(parsePoint)

#学習済みモデルで判定行う用
res = test.map(lambda data: model.predict(data))

# SVMで学習実行
model = SVMWithSGD.train(parsedData, iterations=5000)#SGDで5000回反復

# ---------- Draw Graph ---------- #
plt.figure(figsize=(9,7))

#訓練データ点をプロット
for i, color in enumerate('rb'):
    idx = np.where(data_t[:,2] == i)[0]#ラベル別でインデックス取得
    plt.scatter(data_t[idx,0],data_t[idx,1], c=color, s=30, alpha=.7)#散布図

#分離平面の重みを読み込む
W = model.weights
B = model.intercept

#分離平面を描写
x = np.linspace(0,2,10)
y= -(W[0]/W[1])*x+B
plt.plot(x,y,"r-",linewidth=2)

#テストデータの判定結果のベクトルを取得
A_l = np.array(res.collect())

#テストデータの描写
for i, color in enumerate('rb'):
    #print(np.where(A_l == i))
    idx = np.where(A_l == i)[0]#ラベル別でインデックス取得
    plt.scatter(test_t[idx,0],test_t[idx,1], c=color, marker="*", s=50, zorder=100)#散布図

#図の様式はテキトー
plt.xlim(0.2,1.4)
plt.ylim(0,22)

#保存
#plt.savefig("{}/coin_svm.png".format(P))

plt.show()
