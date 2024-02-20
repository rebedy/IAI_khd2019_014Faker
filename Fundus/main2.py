import os
import argparse
import time
import random
import numpy as np

from keras.utils import np_utils
from keras import optimizers

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from model import cnn_sample
from dataprocessing2 import image_preprocessing, dataset_loader

# setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded')

    def infer(data, rescale=RESCALE, resize_factor=RESIZE):
        # #### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # To preprocess test data as training data
            X.append(image_preprocessing(d, rescale, resize_factor))
        X = np.array(X)

        pred = model.predict_classes(X)  # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_classes', type=int, default=4)  # DO NOT CHANGE num_classes(4)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes

    """ Model """

    learning_rate = 1e-4

    h, w = int(3072 // RESIZE), int(3900 // RESIZE)
    model = cnn_sample(in_shape=(h, w, 3), num_classes=num_classes)
    adam = optimizers.Adam(lr=learning_rate, decay=1e-5)  # optional optimization
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    bind_model(model)
    if config.pause:
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':
        print('Training Start...')

        img_path = DATASET_PATH + '/train'

        # dataset_loader 함수 최적화 필요

        tmp_epoch = 1
        for i in range(10):

            images, labels = dataset_loader(img_path, idx=i, resize_factor=RESIZE, rescale=RESCALE)

            # containing optimal parameters
            dataset = [[X, Y] for X, Y in zip(images, labels)]
            random.shuffle(dataset)
            X = np.array([n[0] for n in dataset])
            Y = np.array([n[1] for n in dataset])

            """ Callback """
            monitor = 'categorical_accuracy'
            reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

            """ Training loop """
            STEP_SIZE_TRAIN = len(X) // batch_size
            print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))
            t0 = time.time()

            train_val_ratio = 0.8
            tmp = int(len(Y) * train_val_ratio)
            X_train = X[:tmp]
            Y_train = Y[:tmp]
            X_val = X[tmp:]
            Y_val = Y[tmp:]

            for epoch in range(nb_epoch * 10):
                t1 = time.time()
                print("### Model Fitting.. ###")
                print('epoch = {} / {}'.format(epoch, nb_epoch))
                print('chaeck point = {}'.format(epoch))

                # for no augmentation case
                hist = model.fit(X_train, Y_train,
                                 validation_data=(X_val, Y_val),
                                 batch_size=batch_size,
                                 # initial_epoch=epoch,
                                 callbacks=[reduce_lr],
                                 shuffle=True
                                 )
                t2 = time.time()
                print(hist.history)
                print('Training time for one epoch : %.1f' % ((t2 - t1)))
                train_acc = hist.history['categorical_accuracy'][0]
                train_loss = hist.history['loss'][0]
                val_acc = hist.history['val_categorical_accuracy'][0]
                val_loss = hist.history['val_loss'][0]

                nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc)
                nsml.save(tmp_epoch)
                tmp_epoch += 1
            print('Total training time : %.1f' % (time.time() - t0))
            # print(model.predict_classes(X))
            del X
            del Y
