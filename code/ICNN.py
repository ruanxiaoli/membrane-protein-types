from keras import utils
from keras.models import Model
from keras.layers import *
from keras.utils import plot_model
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report,recall_score
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

start = time.time()
speed = 7

batch_size = 8
num_classes = 8
epochs = 10000

def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_quared_norm) / (1.0 + s_quared_norm)
    result = scale * x
    return result

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.25
    result = K.sum(y_true * K.square(K.relu(1 - margin - y_pred))
                   + lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    return result

class Capsule(Layer):

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)  # Capsule继承**kwargs参数
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)  # 得到激活函数

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)  # 必须继承Layer的build方法

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                b += K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):  # 自动推断shape
        return (None, self.num_capsule, self.dim_capsule)


def MODEL():
    input_image = Input(shape=(1500, 20, 1))
    x = Conv2D(256, (7, 7), activation='relu')(input_image)
    x = Dropout(0.3)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu')(x)
    x = Dropout(0.3)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.3)(x)
    x = AveragePooling2D((1, 1))(x)
    #new layer=model2 new
    #x = Conv2D(32, (1, 1), activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = AveragePooling2D((1, 1))(x)
    #model3 new
    #x = Conv2D(16, (1, 1), activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = AveragePooling2D((1, 1))(x)

    x = Reshape((-1, 32))(x)  # (None, 100, 128) 相当于前一层胶囊(None, input_num, input_dim)
    capsule = Capsule(num_capsule=8, dim_capsule=32, routings=3, share_weights=True)(x)  # capsule-（None,10, 16)
    capsule = Reshape((-1, 16))(capsule)  # (None, 100, 128) 相当于前一层胶囊(None, input_num, input_dim)
    capsule = Capsule(num_capsule=8, dim_capsule=16, routings=3, share_weights=True)(capsule)  # capsule-（None,10, 16)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(capsule)  # 最后输出变成了10个概率值
    model = Model(inputs=input_image, output=output)
    return model

if __name__ == '__main__':
    # 加载数据
    # x_train = pd.read_csv('PSSM_X_train', delim_whitespace=True, header=None)
    # x_train = x_train.values
    # x_train = x_train.reshape(3249, 1500, 20, 1)
    #
    # x_test = pd.read_csv('PSSM_X_test', delim_whitespace=True, header=None)
    # x_test = x_test.values
    # x_test = x_test.reshape(4333, 1500, 20, 1)
    #
    # y_train = pd.read_csv('PSSM_y_train', delim_whitespace=True, header=None)
    # y_train = y_train.values
    # y_train = y_train.reshape(3249, 1)
    #
    # y_test = pd.read_csv('PSSM_y_test', delim_whitespace=True, header=None)
    # y_test = y_test.values
    # y_test = y_test.reshape(4333, 1)


    #dataset2
    x_train = pd.read_csv('PSSM2_X_train', delim_whitespace=True, header=None)
    x_train = x_train.values
    x_train = x_train.reshape(3073, 1500, 20, 1)

    x_test = pd.read_csv('PSSM2_X_test', delim_whitespace=True, header=None)
    x_test = x_test.values
    x_test = x_test.reshape(3604, 1500, 20, 1)

    y_train = pd.read_csv('PSSM2_y_train', delim_whitespace=True, header=None)
    y_train = y_train.values
    y_train = y_train.reshape(3073, 1)

    y_test = pd.read_csv('PSSM2_y_test', delim_whitespace=True, header=None)
    y_test = y_test.values
    y_test = y_test.reshape(3604, 1)




    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # x_train /= 255
    # x_test /= 255
    # from sklearn.cross_validation import train_test_split

    X, X_val, Y, Y_vall = train_test_split(x_train, y_train, test_size=0.2, random_state=20, stratify=y_train)

    Y = utils.to_categorical(Y, num_classes)
    Y_val = utils.to_categorical(Y_vall, num_classes)

    y_test = utils.to_categorical(y_test, num_classes)

    print('propressing finished')

    # # # 加载模型
    model = MODEL()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0)
    # model.load_weights('weights.best_5')
    model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
    model.summary()

    filepath = '12-28Dataset2ISTOweights.best_5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # 训练
    data_augmentation = True

    model.fit(X,Y,batch_size=batch_size,epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks_list,
            shuffle=True,
            verbose = 2)

    model = MODEL()
    model.load_weights('12-31Final-weights.best_5')

    predict = model.predict(x_test)
    label = []


    Pretrain=model.predict(x_train)
    Pretrainlabel = []
    for i in range(len(y_train)):
        Pretrainlabel.append(np.argmax(Pretrain[i]))
        np.savetxt('1-1Final-weightspretrainData2', Pretrainlabel, delimiter=',')

    # Pretrainlabel.append(Pretrai
    # Pretrainlabel.append(Pretrain)
    # np.savetxt('ICNN_pretrainData1', Pretrainlabel, delimiter=',')


    predict = model.predict(X_val)
    label = []
    label2 = []
    for i in range(len(Y_vall)):
        label.append(np.argmax(predict[i]))

    for i in range (len(Y_vall)):
        label2.append(int(Y_vall[i][0]))
    # np.savetxt('12-28ICNN_prevaliData1', label2, delimiter=',')
    k = 0
    for i in range(len(Y_vall)):
        if label[i] == label2[i]:
            k+=1
    print(k/len(Y_vall))

    predict = model.predict(x_test)
    label = []

    for i in range(len(y_test)):
        label.append(np.argmax(predict[i]))
        np.savetxt('1-1Final-weightspretestData2', label, delimiter=',')

    y_test__ = np.loadtxt('PSSM2_y_test')

    print(np.sum(label == y_test__) / len(y_test__))
    print(recall_score(y_test__, label, average=None))
    print(classification_report(y_test__, label))

    np.savetxt('1-1Final-weights_prob_train2',model.predict(x_train))
    np.savetxt('1-1Final-weights_prob_test2', model.predict(x_test))

    plot_model(model, to_file='model.png', show_shapes=True)

end = time.time()
print ("time is the ",end-start)