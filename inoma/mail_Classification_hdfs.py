#!/usr/bin/env python
# coding:utf-8

"""
メール分類用プログラム
データをhdfsから読み込む
"""

import os
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

#P = os.path.abspath(os.path.dirname(__file__))# スクリプトのあるディレクトリへの絶対パス取得
sc = SparkContext()

#データ読み込み
#spam = np.loadtxt('{}/spam.csv'.format(P),delimiter=',')
#spam = sc.parallelize(spam)

#ham = np.loadtxt('{}/ham.csv'.format(P),delimiter=',')
#ham = sc.parallelize(ham)

spam = sc.textFile("hdfs:///spark/spam.csv")
ham  = sc.textFile("hdfs:///spark/ham.csv")

def parsePoint(vec):
    return LabeledPoint(vec[-1], vec[0:-1])

def forfloat(arr):
    tmp = []
    for x in arr:
        tmp.append(float(x))
    return tmp

#RDDデータの振り分け?
spamData = spam.map(lambda qwe : qwe.split(',')).map(lambda x : forfloat(x)).map(parsePoint)
hamData = ham.map(lambda qwe : qwe.split(',')).map(lambda x : forfloat(x)).map(parsePoint)

#訓練データ、テストデータに分ける
weights = [0.7,0.3]#訓練:テスト　の比率
seed = 50 #これ良くわからん
sptrainData, sptestData = spamData.randomSplit(weights, seed)
hatrainData, hatestData = hamData.randomSplit(weights, seed)

#spam,hamの訓練データ、テストデータをひとつに
train = sptrainData.union(hatrainData)
test = sptestData.union(hatestData)
train.cache()#訓練データをキャッシュ

# a = test.take(1)
# print(a)

#学習した後の判別用
res = test.map(lambda data: model.predict(data.features))

#モデルを学習
#model = LogisticRegressionWithLBFGS.train(train,10)#ロジスティック回帰(L-BFGS)で
model = LogisticRegressionWithLBFGS.train(train,25,regParam=0.0002,regType="l1")
#model = SVMWithSGD.train(train,5000,regParam=0.005,regType="l2")#SVM(SGD)で
#model = NaiveBayes.train(train,1.)#ナイベで

#print(len(np.array(model.weights)))

#テストデータの推定結果を取得
Re_l = res.collect()

#評価用に(推定ラベル,正解ラベル)のペアのタプルが並んだリストを生成
#hako = []
#for i,v in enumerate(test.collect()):
#    hako.append((float(Re_l[i]), v.label))
#
#そのリストをspark用に変換,メトリクスのクラスに渡す
predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
##7_12変更
metrics = MulticlassMetrics(predictionAndLabels)

metrics2 = BinaryClassificationMetrics(predictionAndLabels)


#メトリクスのメソッドを使って再現率と適合率を計算＆出力
print("-------------------------------------")
print("under PR = {}".format(metrics2.areaUnderPR))
#print("under ROC = {}".format(metrics2.areaUnderROC))

print("precision of ham = {}".format(metrics.precision(0.)))
print("precision of spam = {}".format(metrics.precision(1.)))
print("recall of ham = {}".format(metrics.recall(0.)))#スパムでないメールの認識率
print("recall of spam = {}".format(metrics.recall(1.)))#スパムメールの検出率

sc.stop()
