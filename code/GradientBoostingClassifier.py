from xgboost import XGBClassifier
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


    # learnrate =range(0.05,0.3,0.020)
    # maxdepth =range(1,10,1)

    # learnrate =[0.005,0.007,0.009,0.11,0.14,0.17,0.21,0.24,0.27,0.3]
    # estimators =range(1,10,1)


    estimators =range(10,110,10)
    learnrate = [0.005,0.007,0.009,0.11,0.14,0.17,0.21,0.24,0.27,0.3]

    bestestimators = 0
    bestlearnrate = 0
    bestScore = 0
    for i in estimators:
        for j in learnrate:
            sum = 0
            for k in range(m):
                trainData0, trainLabel0, testData, testLabel = jackknife(trainData, trainLabel, k)
                nortrain=Normalizer().fit_transform(trainData0)
                nortest = Normalizer().fit_transform(testData.reshape(1, -1))
                #默认参数：n_estimators=100,max_features=None,max_depth=3,min_sample_split=2,min_sample_leaf=1,subsample=1,
                rf1=GradientBoostingClassifier(n_estimators=i, learning_rate=j,loss='deviance',subsample=0.5,random_state=1000)#min_samples_split为总样本的0.5%-1%。
                rf1.fit(nortrain, trainLabel0)
                if testLabel == np.squeeze(rf1.predict(nortest.reshape(1, -1))):
                    sum += 1
            score = sum / len(trainLabel)
            print(score)
            if score > bestScore:
                bestScore = score
                bestestimators = i
                bestlearnrate = j
    return  bestestimators,bestlearnrate

dataset=np.loadtxt("494Pssmcc-9.csv",delimiter=',')
trainMat = dataset[:,0:3620]
trainLabel = dataset[:,3620]

bestestimators,bestlearnrate = KNN(trainMat, trainLabel)
print(bestestimators,bestlearnrate)
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
    rf1 = GradientBoostingClassifier(loss='deviance',n_estimators=bestestimators, learning_rate=bestlearnrate,subsample=0.5,random_state=1000)#loss: exponential;subsample[0.5, 0.8],n_estimators默认为100
    rf1.fit(nortrain, trainLabel0)
    res.append(rf1.predict(nortest.reshape(1, -1)))
    newtrain1 = rf1.predict_proba(nortest)
    newtrain[i]=rf1.predict_proba(nortest)
    pro.append(newtrain[i])
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/GBC-494Pssmcc-9pre.csv', res, delimiter=',')
    np.savetxt('F:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/GBC-494Pssmcc-9pro.csv', pro,delimiter=',')
overall_accuracy = accuracy_score(y_true = trainLabel, y_pred = res)#label  predict
recall=metrics.recall_score(trainLabel,res,average=None)
MCC=matthews_corrcoef(trainLabel,res)
m_precision= metrics.precision_score(y_true= trainLabel, y_pred= res,average=None)

print('OA',overall_accuracy)
print('m_precision',m_precision)
print('recall',recall)
print('MCC',MCC)