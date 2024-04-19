import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from imblearn.combine import SMOTEENN
from minepy import MINE
import keras.backend as k
k.clear_session()

seed=7

'''
# KNN
def jackknife(trainData, trainLabel, k):
    testData = trainData[k]
    testLabel = trainLabel[k]
    trainData = list(trainData)
    trainLabel = list(trainLabel)
    del (trainData[k])
    del (trainLabel[k])
    return trainData, trainLabel, testData, testLabel

def KNN(trainData, trainLabel):
    m = len(trainData)
    k_range = range(1, 100)
    bestK = 1
    bestScore = 0
    for i in k_range:
        sum = 0
        for k in range(m):
            trainData0, trainLabel0, testData, testLabel = jackknife(trainData, trainLabel, k)
            nortrain = Normalizer().fit_transform(trainData0)
            nortest = Normalizer().fit_transform(testData.reshape(1, -1))
            rf1 = KNeighborsClassifier(n_neighbors=i)
            rf1.fit(nortrain, trainLabel0)
            if testLabel == np.squeeze(rf1.predict(nortest.reshape(1, -1))):
                sum += 1
        score = sum / len(trainLabel)
        print(score)
        if score > bestScore:
            bestScore = score
            bestK = i
    return bestK

dataset=np.loadtxt("pseacc983-2-30.csv",delimiter=',')
trainMat = dataset[:,0:200]
trainLabel = dataset[:,200]

# sm=SMOTEENN()
# trainMat, trainLabel = sm.fit_sample(trainMat, trainLabel)
bestK = KNN(trainMat, trainLabel)
print(bestK)
res = []
pro= []
ntrain = trainMat.shape[0]
classnum = len(np.unique(trainLabel))
newtrain= np.zeros((ntrain, classnum))
for i in range(len(trainMat)):
    trainData0, trainLabel0, testData, testLabel = jackknife(trainMat, trainLabel, i)
    nortrain = Normalizer().fit_transform(trainData0)
    nortest = Normalizer().fit_transform(testData.reshape(1, -1))
    rf1 = KNeighborsClassifier(n_neighbors=bestK)
    rf1.fit(nortrain, trainLabel0)
    res.append(rf1.predict(nortest.reshape(1, -1)))
    newtrain1 = rf1.predict_proba(nortest)
    newtrain[i] = rf1.predict_proba(nortest)
    pro.append(newtrain[i])
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/thirdly/classification/classify/KNNpseacc983-2-30.csv', res, delimiter=',')
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/thirdly/classification/classify/KNNpseacc983-2-30pro.csv', pro,delimiter=',')
overall_accuracy = accuracy_score(y_true = trainLabel, y_pred = res)#label  predict
recall=metrics.recall_score(trainLabel,res,average=None)
MCC=matthews_corrcoef(trainLabel,res)
m_precision= metrics.precision_score(y_true= trainLabel, y_pred= res,average=None)
print('OA',overall_accuracy)
print('m_precision',m_precision)
print('recall',recall)
print('MCC',MCC)
'''

# '''
# RF https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/ATC_RF_Kfold.py
#默认参数：max_features=auto,max_depth=None,min_samples_split=2,min_samples_leaf=1,
def jackknife(trainData, trainLabel, k):
    testData = trainData[k]
    testLabel = trainLabel[k]
    trainData = list(trainData)
    trainLabel = list(trainLabel)
    del (trainData[k])
    del (trainLabel[k])
    return trainData, trainLabel, testData, testLabel
def KNN(trainData, trainLabel):
    m = len(trainData)
    # max_features=[]
    # estimators=[5,10]#一般设置为100
    # estimators=range(10,110,10)
    # maxdepth=range(1,100,5)

    estimators =range(1,110,10)
    maxdepth =range(1,10,1)

    bestestimators = 0
    bestmaxdepth = 0
    bestScore = 0
    for i in estimators:
        for j in maxdepth:
            sum = 0
            for k in range(m):
                trainData0, trainLabel0, testData, testLabel = jackknife(trainData, trainLabel, k)
                nortrain=Normalizer().fit_transform(trainData0)
                nortest = Normalizer().fit_transform(testData.reshape(1, -1))
                rf1 = RandomForestClassifier(n_estimators=i,max_depth=j,min_samples_leaf=1,random_state=1000)# max_feature:auto,None,sqrt,log2,int:整数;n_estimators默认为10
                rf1.fit(nortrain, trainLabel0)
                if testLabel == np.squeeze(rf1.predict(nortest.reshape(1, -1))):
                    sum += 1
            score = sum / len(trainLabel)
            print(score)
            if score > bestScore:
                bestScore = score
                bestestimators = i
                bestmaxdepth = j
    return bestestimators, bestmaxdepth

# dataset=np.loadtxt("ALLM317QZH50-75-25ACC.csv",delimiter=',')
# trainMat = dataset[:,0:200]
# trainLabel = dataset[:,200]


dataset=np.loadtxt("ALLM1217QZH50-75-25ACC.csv",delimiter=',')
trainMat = dataset[:,0:200]
trainLabel = dataset[:,200]

bestestimators, bestmaxdepth = KNN(trainMat, trainLabel)
print(bestestimators, bestmaxdepth)
res = []
pro= []
ntrain = trainMat.shape[0]
classnum = len(np.unique(trainLabel))
newtrain= np.zeros((ntrain, classnum))
print(newtrain.shape)
for i in range(len(trainMat)):
    trainData0, trainLabel0, testData, testLabel = jackknife(trainMat, trainLabel, i)
    nortrain = Normalizer().fit_transform(trainData0)
    nortest = Normalizer().fit_transform(testData.reshape(1, -1))
    rf1 = RandomForestClassifier(n_estimators=bestestimators,max_depth=bestmaxdepth,min_samples_leaf=1,random_state=1000)
    rf1.fit(nortrain, trainLabel0)
    res.append(rf1.predict(nortest.reshape(1, -1)))
    newtrain1 = rf1.predict_proba(nortest)
    newtrain[i]=rf1.predict_proba(nortest)
    pro.append(newtrain[i])
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/RFALLM1217QZH50-75-25ACCpre.csv', res, delimiter=',')
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/RFALLM1217QZH50-75-25ACCpro.csv', pro,delimiter=',')
overall_accuracy = accuracy_score(y_true = trainLabel, y_pred = res)#label  predict
recall=metrics.recall_score(trainLabel,res,average=None)
MCC=matthews_corrcoef(trainLabel,res)
m_precision= metrics.precision_score(y_true= trainLabel, y_pred= res,average=None)

print('OA',overall_accuracy)
print('m_precision',m_precision)
print('recall',recall)
print('MCC',MCC)

