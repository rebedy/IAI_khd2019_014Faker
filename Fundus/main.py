import os
import argparse
import time
import cv2
import numpy as np
import pickle
from subprocess import call

from keras.utils import np_utils
from keras import optimizers

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

from model import cnn_sample
# from model import InceptionV3
from config_setting import *
from dataprocessing import *


# setting values of preprocessing parameters
RESIZE = 10.
RESCALE = False


def bind_model(model, aug_mean_image):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded')

    def infer(data, resize_factor=RESIZE):  # test mode
        # #### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            h, w, _ = 3072, 3900, 3
            nh, nw = int(h // resize_factor), int(w // resize_factor)
            res = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_AREA)
            X.append(res)

            # X.append(image_preprocessing(d, resize_factor))
        X = np.array(X) - aug_mean_image

        pred = model.predict_classes(X)  # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=50)  # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)  # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)  # DO NOT CHANGE num_classes, class 수는 항상 4

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

    if config.pause:
        print('Inferring Start...')
        nsml.paused(scope=locals())

    for count_kfold in range(NUMBER_OF_FOLD):
        if config.mode == 'train':
            print('Training Start...')

            """ Augmentation """
            fold_dir = os.path.join(KFOLD_DIR, '%sfold' % str(count_kfold + 1))
            call('pwd')
            call(['ls', KFOLD_DIR])

            fold_aug_train_dir = os.path.join(fold_dir, 'train_aug')
            fold_aug_valid_dir = os.path.join(fold_dir, 'valid_aug')
            fold_mean_sub = os.path.join(fold_dir, 'fold%s_aug_mean_image.pickle' % str(count_kfold + 1))
            print("===========\n")
            call(['ls', fold_dir])

            train_path_list, valid_path_list, aug_train_mean_image_path = [], [], 0
            for Par_dir, Sub_name, File_name in os.walk(fold_dir):
                if File_name:
                    for itr in range(len(File_name)):
                        file_path = os.path.join(Par_dir, File_name[itr])
                        if '_t.pickle' in file_path:
                            print(file_path)
                            train_path_list.append(file_path)
                        elif '_valid.pickle' in file_path:
                            valid_path_list.append(file_path)
                        elif 'aug_mean_image.pickle' in file_path:
                            aug_train_mean_image_path = file_path
                            print("===========\n")
                            print(aug_train_mean_image_path)
                        else:
                            aug_train_mean_image_path = None
                            pass

            valid_x, valid_y = [], []
            for i, file_itr in enumerate(valid_path_list):
                with open(str(file_itr), 'rb') as f:
                    valid = pickle.load(f)
                    np_array_values = valid[0:-1]
                    get_label = valid[-1]
                    valid_array = np_array_values[0]
                    label = np_utils.to_categorical(get_label, 4)
                    valid_x = np.array(valid_array)
                    valid_y = np.array(label)

            with open(str(aug_train_mean_image_path), 'rb') as f:
                aug_mean_image = pickle.load(f)
                # print('aug_mean_image: ',aug_mean_image)
                aug_mean_image = np.array(aug_mean_image[0])
                print('squeeze 한거 임. aug_mean_image: ', aug_mean_image.shape)

            for i, file_itr in enumerate(train_path_list):
                with open(str(file_itr), 'rb') as f:
                    train = pickle.load(f)  # pickle.dump([aug_array, aug_label, aug_name], f) #type : list
                    np_array_values = train[0:-1]
                    # print("np_array_values : ", np_array_values)
                    # print('np_array_values: ', np_array_values
                    # )
                    get_label = train[-1]
                    train_array = np_array_values[0]
                    # print('train_array      ', train_array)
                    label = np_utils.to_categorical(get_label, 4)  # keras.utils.
                    X = np.array(train_array)
                    Y = np.array(label)
                    # print("X : ", X.shape)

                    # for i, timg in enumerate(X):
                    train_X = X - aug_mean_image
                    # print("X_2 : ", train_X.shape)
                    # for i, vimg in enumerate(valid_x):
                    valid_x = valid_x - aug_mean_image

                    train_X = np.array(train_X)
                    valid_x = np.array(valid_x)

                    print(train_X.shape, valid_x.shape)
                    h, w = int(3072 // RESIZE), int(3900 // RESIZE)
                    model = cnn_sample(in_shape=(h, w, 3), num_classes=num_classes)

                    adam = optimizers.Adam(lr=learning_rate, decay=1e-5)  # optional optimization
                    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
                    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

                    bind_model(model, aug_mean_image)

                    """ Callback """
                    monitor = 'categorical_accuracy'
                    # reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
                    """"""

                    """ Training loop """
                    STEP_SIZE_TRAIN = len(train_X) // batch_size
                    print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))
                    t0 = time.time()

                    nsml.load(checkpoint='28', session='team017/KHD2019_FUNDUS/116')
                    nsml.save('saved')

                    for epoch in range(nb_epoch):
                        t1 = time.time()
                        print("### Model Fitting.. ###")
                        print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
                        print('check point = {}'.format(epoch))

                        print(train_X.shape)
                        print(Y.shape)
                        # for no augmentation case
                        hist = model.fit(train_X, Y, validation_data=(valid_x, valid_y), batch_size=8)

                        t2 = time.time()
                        print(hist.history)
                        print('Training time for one epoch : %.1f' % ((t2 - t1)))
                        train_acc = hist.history['categorical_accuracy'][0]
                        train_loss = hist.history['loss'][0]
                        val_acc = hist.history['val_categorical_accuracy'][0]
                        val_loss = hist.history['val_loss'][0]

                        nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc,
                                    val_loss=val_loss, val_acc=val_acc)
                        nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
