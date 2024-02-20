import os
from os.path import join
import time
import random
import cv2
import numpy as np
import pickle
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedKFold

from nsml.constants import DATASET_PATH, GPU_NUM

from config_setting import *


def Label2Class(label):  # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    res = []
    if label == 'AMD':
        cls = 1;
        res.append(cls)
        # resvec[cls] = 1
    elif label == 'RVO':
        cls = 2;
        res.append(cls)
        # resvec[cls] = 1
    elif label == 'DMR':
        cls = 3;
        res.append(cls)
        # resvec[cls] = 1
    else:
        cls = 0;
        res.append(cls)
        # resvec[cls] = 1  # Normal

    return res, resvec


def dataset_loader(img_path, resize_factor):
    t1 = time.time()
    print('Loading training data...\n')
    print('Starts @ ', t1)

    # 1. List of image path
    p_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(img_path) for f in files if
              all(s in f for s in ['.jpg'])]
    p_list.sort()
    num_data = len(p_list)

    # 2. Adjusting image size and pixel range
    h, w, _ = 3072, 3900, 3
    nh, nw = int(h // resize_factor), int(w // resize_factor)

    img_array, labels, _ = [], [], []
    for i, p in enumerate(p_list):
        temp_arr = np.array(Image.open(p))
        temp_arr = temp_arr[int(temp_arr.shape[0] * 0.15):int(temp_arr.shape[0] * 0.85), int(temp_arr.shape[1] * 0.15): int(temp_arr.shape[1] * 0.85)]
        open_and_resize = cv2.resize(temp_arr, (nw, nh), interpolation=cv2.INTER_AREA)  # interpolation=

        img_array.append(np.array(open_and_resize))
        # 3, Generate label data
        l, _ = Label2Class(p.split('/')[-2])
        labels.append(l)

        print(i + 1, '/', num_data, ' image(s)')
    images = np.array(img_array)
    labels = np.array(labels)

    t2 = time.time()
    print('Dataset prepared for', t2 - t1, 'sec')
    print('Images:', images.shape, 'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, ' among 0-3 classes')

    return p_list, images, labels


def AugmentationCombination(itr, train_array, train_label, train_name, LR_reverse_option, adjust_brightness_option,
                            control_intensity_option):
    img_temp = train_array
    label_temp = train_label
    name_temp = train_name

    if LR_reverse_option['LR_reverse']:
        reverse_or_not = LR_reverse_option['LR_case'][np.random.randint(len(LR_reverse_option['LR_case']))]
        reversed_img_set, reversed_img_names = [], []
        for i, reverse_itr in enumerate(img_temp):
            reversed_img = left_right_reverse(reverse_itr, reverse_or_not=reverse_or_not)
            reversed_img_set.append(np.array(reversed_img))
            reversed_img_names.append(name_temp[i] + '_' + str(itr))
        img_temp = np.array(reversed_img_set)
        name_temp = np.array(reversed_img_names)
    else:
        pass

    # ## 2. Adjust_brightness ##
    if adjust_brightness_option['adjust_brightness']:
        delta_values = list(adjust_brightness_option['delta'])
        index = 0
        adjusted_brightness_img, adjusted_brightness_name = [], []
        # for i in range(5):
        #     print(img_temp[i].shape,'\n')
        for i, brightness_itr in enumerate(img_temp):
            selected_delta_value = random.sample(delta_values, 1)
            # print('        selected_delta_value for %dth img: '%index,selected_delta_value)
            adjust_brightness_img = img_bright_adjust(brightness_itr, selected_delta_value)
            adjusted_brightness_img.append(np.array(adjust_brightness_img))
            index += 1
            adjusted_brightness_name.append(name_temp[i] + '_' + str(itr))
        name_temp = np.array(adjusted_brightness_name)
        img_temp = np.array(adjusted_brightness_img)
    else:
        pass

    # ## 3. Add intensity ##
    # print("   >>> Adding Intensity")
    if control_intensity_option['control_intensity']:
        values_to_add = list(control_intensity_option['values'])
        # added_img = np.empty(shape=img_temp.shape)
        added_pixel_intensity, intensity_name = [], []
        index = 0
        # for i in range(5):
        #     print(img_temp[i].shape,'\n')
        for i, each_img in enumerate(img_temp):
            selected_value = np.array(random.sample(values_to_add, 1)[0])
            added_pixel_intensity.append(add_pixel_intensity(each_img, selected_value))
            # augmented_label.append()
            index += 1
            intensity_name.append(name_temp[i] + '_' + str(itr))
        name_temp = np.array(intensity_name)
        img_temp = np.array(added_pixel_intensity)
    else:
        pass

    # ## 4. mean image cal ##
    mean_image = 0
    for itr in img_temp:
        mean_image += itr
    mean_image = mean_image.astype('float32')
    mean_image /= len(img_temp)

    return np.array(img_temp), label_temp, name_temp, np.array(mean_image)


def left_right_reverse(image, reverse_or_not):
    if reverse_or_not == 0:
        left_right_reversed_img = image
    else:
        left_right_reversed_img = np.fliplr(image)
        # left_right_reversed_img = image[:,::-1,:]
    return left_right_reversed_img


def img_bright_adjust(img, value):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    adjusted_img_temp = ImageEnhance.Brightness(img)
    adjusted_img = adjusted_img_temp.enhance(np.array(value))
    adjusted_img = np.array(adjusted_img)
    return adjusted_img


def add_pixel_intensity(img, value):
    img_pixel_array = np.copy(img)

    if img.dtype == np.uint16:
        if value >= 0:
            overflow_place = (img_pixel_array >= (2 ** 16 - value - 1))
            normal_place = np.invert(overflow_place)

            img_pixel_array[overflow_place] = 2 ** 16 - 1
            img_pixel_array[normal_place] = img_pixel_array[normal_place] + value

        elif value < 0:
            underflow_place = (img_pixel_array < np.abs(value))
            normal_place = np.invert(underflow_place)
            img_pixel_array[underflow_place] = 0
            img_pixel_array[normal_place] = img_pixel_array[normal_place] + value
        else:
            pass

    elif img.dtype == np.uint8:
        if value >= 0:
            overflow_place = (img_pixel_array > (2 ** 8 - value - 1))
            normal_place = np.invert(overflow_place)

            img_pixel_array[overflow_place] = (2 ** 8 - 1)
            img_pixel_array[normal_place] = img_pixel_array[normal_place] + value

        elif value < 0:  # value= -1 ~ n (where n >> -inf)
            underflow_place = (img_pixel_array < np.abs(value))
            normal_place = np.invert(underflow_place)
            img_pixel_array[underflow_place] = 0
            img_pixel_array[normal_place] = img_pixel_array[normal_place] + value
    else:
        print('img type is wrong. please check!')
        raise ValueError

    return np.array(img_pixel_array)


""" Directory Setting """
img_path = DATASET_PATH + '/train/'
img_dir_list, images, label_list = dataset_loader(img_path, resize_factor=RESIZE)  # , rescale=RESCALE)
# # containing optimal parameters

""" Stratified KFOLD """
count_kfold = 1
data_fold_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
ALPHA = 5

for train_idx, valid_idx in data_fold_splitter.split(X=img_dir_list, y=label_list):
    idx_step = int(len(train_idx) / ALPHA)
    print("len(train_idx): ", len(train_idx))
    print('idx_step: ', idx_step)
    real_mean_image = 0

    # make directory
    # fold_dir =os.path.join(KFOLD_DIR, '%sfold' % str(count_kfold + 1))
    fold_dir = make_dir_ifnot(os.path.join(KFOLD_DIR, '%sfold' % str(count_kfold)))  # fold 만들어짐
    print("+====data prepcessing====")
    call('pwd')
    call(['ls', KFOLD_DIR])

    fold_aug_train_dir = make_dir_ifnot(os.path.join(fold_dir, 'train_aug'))
    fold_aug_valid_dir = make_dir_ifnot(os.path.join(fold_dir, 'valid_aug'))
    fold_mean_sub = os.path.join(fold_dir, 'fold%s_aug_mean_image.pickle' % str(count_kfold))
    print("===========\n")
    call(['ls', fold_dir])

    """ Augmentation """
    # fold_dir = make_dir_ifnot(join(KFOLD_DIR, '%sfold' % str(count_kfold)))
    # fold_aug_train_dir = make_dir_ifnot(join(fold_dir, 'train_aug'))
    # fold_aug_valid_dir = make_dir_ifnot(join(fold_dir, 'valid_aug'))

    for idx in range(ALPHA):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if idx+1 == ALPHA:
            train_x = images[(idx+1) * idx_step:]
            train_y = label_list[(idx+1) * idx_step:]
            train_name = [img_name.split('/')[-1][:-4] for img_name in img_dir_list[(idx+1) * idx_step:]]

        else:
            train_x = images[idx * idx_step: (idx + 1) * idx_step]
            train_y = label_list[idx * idx_step: (idx + 1) * idx_step]
            train_name = [img_name.split('/')[-1][:-4]
                          for img_name in img_dir_list[idx * idx_step: (idx + 1) * idx_step]]

        # print("train_x.shape",np.array(train_x).shape)
        # print("train_y.shape",np.array(train_y).shape)

        temp_mean_image = 0
        for ever in range(multiply_by_aug):
            aug_array, aug_label, aug_name, pixel_wise_mean_image = \
                AugmentationCombination(ever, np.array(train_x), train_y, train_name,  # train_batch_age,train_batch_se,
                                        LR_reverse_option=train_LR_reverse_option,
                                        adjust_brightness_option=train_adjust_brightness_option,
                                        control_intensity_option=train_control_intensity_option)
            # print('pixel_wise_mean_image  ', pixel_wise_mean_image.shape)
            # print('pixel_wise_mean_image  ', type(pixel_wise_mean_image))
            temp_mean_image += pixel_wise_mean_image
            print('type(temp_mean_image: ', type(temp_mean_image))

            aug_array = np.array(aug_array)
            aug_label = np.array(aug_label)
            print("aug_array.shape", aug_array.shape)
            # print('aug_array: ', aug_array)
            # print('aug_label: ', aug_label)
            """ Dumping Augmented Pickle file """
            fold_aug_train_pickle_file = join(fold_aug_train_dir,
                                              'fold%d_train_%d_aug_%d_t.pickle' % (count_kfold, idx, ever))
            with open(fold_aug_train_pickle_file, 'wb') as f:
                pickle.dump([aug_array, aug_label], f)

        temp_mean_image = temp_mean_image.astype('float32') / int(multiply_by_aug)
        real_mean_image += temp_mean_image
    """ Saving Pixel-Wise Mean Image """
    aug_mean_image = real_mean_image.astype('float32') / int(ALPHA)
    # scipy.misc.imsave(join(fold_dir, 'fold%d_aug_mean_image.jpg' % count_kfold), aug_mean_image)

    with open((fold_dir + '/' + 'fold%d_aug_mean_image.pickle' % count_kfold), 'wb') as f:
        pickle.dump([aug_mean_image], f)

    print('len(aug_mean_image): ', np.array(aug_mean_image).shape)

    """ Valid Set """
    valid_x = [images[j] for j in valid_idx]
    valid_y = [label_list[j] for j in valid_idx]
    valid_name = [img_dir_list[j].split('/')[-1][:-4] for j in valid_idx]
    fold_aug_valid_pickle_file = join(fold_aug_valid_dir, 'fold%d_valid.pickle' % count_kfold)
    with open(fold_aug_valid_pickle_file, 'wb') as f:
        pickle.dump([valid_x, valid_y], f)
        # pickle.dump([valid_x, valid_y, valid_name], f)

    count_kfold += 1
