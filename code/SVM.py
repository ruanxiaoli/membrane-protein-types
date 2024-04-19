import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn import datasets, svm
import keras.backend as k
k.clear_session()
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    C = [0.001,0.03,0.1, 0.3, 1, 3, 10,30,100]
    gamma = [0.0003,0.001, 0.003, 0.01, 0.03, 0.1,0.3,1,3,10,30,40]

    bestC = 0
    bestGamma = 0
    bestScore = 0
    for i in C:
        for j in gamma:
            sum = 0
            for k in range(m):
                trainData0, trainLabel0, testData, testLabel = jackknife(trainData, trainLabel, k)

                nortrain=Normalizer().fit_transform(trainData0)
                nortest = Normalizer().fit_transform(testData.reshape(1, -1))

                rf1 = svm.SVC(kernel='rbf', C=i, gamma=j)
                # rf1 = KNeighborsClassifier(n_neighbors=1)
                rf1.fit(nortrain, trainLabel0)

                if testLabel == np.squeeze(rf1.predict(nortest.reshape(1, -1))):
                    sum += 1
            score = sum / len(trainLabel)
            print(score)
            if score > bestScore:
                bestScore = score
                bestC = i
                bestGamma = j
    return bestC, bestGamma


dataset=np.loadtxt("1217DDE.csv",delimiter=',')
trainMat = dataset[:,0:400]
trainLabel = dataset[:,400]


bestC, bestGamma = KNN(trainMat, trainLabel)
print(bestC, bestGamma)
clf = svm.SVC(kernel='rbf', C=bestC, gamma=bestGamma)
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

    rf1 = svm.SVC(kernel='rbf', C=bestC, gamma=bestGamma,probability=True)
    # rf1 = KNeighborsClassifier(n_neighbors=1)
    rf1.fit(nortrain, trainLabel0)
    res.append(rf1.predict(nortest.reshape(1, -1)))

    newtrain[i]=rf1.predict_proba(nortest.reshape(1,-1))


    pro.append(newtrain[i])
    # print(pro)
    #
    np.savetxt('D:/pycharm\PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/1217DDEpre.csv', res, delimiter=',')
    np.savetxt('D:/pycharm\PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/1217DDEpro.csv', pro,delimiter=',')



overall_accuracy = accuracy_score(y_true = trainLabel, y_pred = res)#label  predict
recall=metrics.recall_score(trainLabel,res,average=None)
MCC=matthews_corrcoef(trainLabel,res)
m_precision= metrics.precision_score(y_true= trainLabel, y_pred= res,average=None)

print('OA',overall_accuracy)
print('m_precision',m_precision)
print('recall',recall)
print('MCC',MCC)



