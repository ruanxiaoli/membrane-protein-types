import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import datasets, svm
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# import keras.backend as k
# k.clear_session()


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
    K=range(1, 100, 1)

    bestk = 0
    # bestGamma = 0
    bestScore = 0
    for i in K:
        sum = 0
        for k in range(m):
            trainData0, trainLabel0, testData, testLabel = jackknife(trainData, trainLabel, k)

            nortrain = Normalizer().fit_transform(trainData0)
            nortest = Normalizer().fit_transform(testData.reshape(1, -1))

            # rf1 = svm.SVC(kernel='rbf', C=i, gamma=j)
            rf1 = KNeighborsClassifier(n_neighbors=i)
            rf1.fit(nortrain, trainLabel0)

            if testLabel == np.squeeze(rf1.predict(nortest.reshape(1, -1))):
                sum += 1
        score = sum / len(trainLabel)
        print(score)
        if score > bestScore:
            bestScore = score
            bestK = i
            # bestGamma = j

    return bestK


dataset=np.loadtxt("1217DDE.csv",delimiter=',')
trainMat = dataset[:,0:400]
trainLabel = dataset[:,400]


bestK = KNN(trainMat, trainLabel)
print(bestK)
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

    rf1 = KNeighborsClassifier(n_neighbors=bestK)
    rf1.fit(nortrain, trainLabel0)
    res.append(rf1.predict(nortest.reshape(1, -1)))

    newtrain[i]=rf1.predict_proba(nortest.reshape(1,-1))


    pro.append(newtrain[i])
    #
    np.savetxt('D:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/1217DDE-2pre.csv', res, delimiter=',')
    np.savetxt('D:/pycharm/PyCharm 2017.3.4/PycharmProjects/graduationchaper4/classification/1217DDE-2pro.csv', pro,delimiter=',')



overall_accuracy = accuracy_score(y_true = trainLabel, y_pred = res)#label  predict
recall=metrics.recall_score(trainLabel,res,average=None)
MCC=matthews_corrcoef(trainLabel,res)
m_precision= metrics.precision_score(y_true= trainLabel, y_pred= res,average=None)

print('OA',overall_accuracy)
print('m_precision',m_precision)
print('recall',recall)
print('MCC',MCC)



