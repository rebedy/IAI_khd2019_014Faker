import os
import numpy as np
from nsml.constants import DATASET_PATH, GPU_NUM


def make_dir_ifnot(path):
    try:
        os.makedirs(path, exist_ok=True) # python >= 3.2
        # path = Path(path)
        # path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # Python > 2.5
        raise

    return path


KFOLD_DIR = make_dir_ifnot('/app/')
# KFOLD_DIR = '/app/KDH2019_UWF_fundus_Aug_dataset_team_Faker'
NUMBER_OF_FOLD = 3
RESIZE = 10.
# RESCALE = False #### 안써!!!!
ALPHA = 5

""" Augmentation Options """
# "Crop"
# value_of = [(0, 0), (0, 12), (12, 0), (6, 6), (12, 12)]  # 5배

# "LR_Flip"
LR_reverse_option = {'LR_reverse': True}
train_LR_reverse_option = {'LR_reverse': True, 'LR_case': [0, 1], 'name': 'LR_reverse'}

# "Intensity"
add_intensity_values = np.arange(0, 11, step=5)  # (1, 6) # (-5,5)
train_control_intensity_option = {'control_intensity': True, 'values': add_intensity_values, 'name': 'intensity'}

# "Bright"
brightness_range = (0.8, 1.2)
img_brightness_delta = np.linspace(brightness_range[0], brightness_range[-1], num=2, endpoint=True, dtype='float')
train_adjust_brightness_option = {'adjust_brightness': True, 'delta': img_brightness_delta}

""" Calculating records for Augmentation!!! """
multiply_by_aug = 1  # len_train_data
no_train_records = 1
if train_LR_reverse_option['LR_reverse']:
    multiply_by_aug *= len(train_LR_reverse_option['LR_case'])
else:
    pass
if train_control_intensity_option['control_intensity']:
    multiply_by_aug *= len(train_control_intensity_option['values'])
else:
    pass
if train_adjust_brightness_option['adjust_brightness']:
    multiply_by_aug *= len(train_adjust_brightness_option['delta'])
else:
    pass


def make_dir_ifnot(path, type=0):
    try:
        if not os.path.exists(path):
            if type == 0:
                os.mkdir(os.path.join(path))
            elif type == 1:
                os.makedirs(os.path.join(path))
            else:
                print("Please check the 'type' arg; ", type, "\n")

        else:
            pass
        # print(path, " is already exist.\n")
    except OSError as e:
        print(e.errno)
        print("DY: Failed to create directories, %s" % path)
        raise
    return path


""" Augmentation Options """
# "Crop"
# value_of = [(0, 0), (0, 12), (12, 0), (6, 6), (12, 12)]  # 5배

# "LR_Flip"
LR_reverse_option = {'LR_reverse': True}
train_LR_reverse_option = {'LR_reverse': True, 'LR_case': [0, 1], 'name': 'LR_reverse'}

# "Intensity"
add_intensity_values = np.arange(0, 6, step=5)  # (1, 6) # (-5,5)
train_control_intensity_option = {'control_intensity': True, 'values': add_intensity_values, 'name': 'intensity'}

# "Bright"
brightness_range = (0.8, 1.2)
img_brightness_delta = np.linspace(brightness_range[0], brightness_range[-1], num=2, endpoint=True, dtype='float')
train_adjust_brightness_option = {'adjust_brightness': True, 'delta': img_brightness_delta}

""" Calculating records for Augmentation!!! """
multiply_by_aug = 1  # len_train_data
no_train_records = 1
if train_LR_reverse_option['LR_reverse']:
    multiply_by_aug *= len(train_LR_reverse_option['LR_case'])
else:
    pass
if train_control_intensity_option['control_intensity']:
    multiply_by_aug *= len(train_control_intensity_option['values'])
else:
    pass
if train_adjust_brightness_option['adjust_brightness']:
    multiply_by_aug *= len(train_adjust_brightness_option['delta'])
else:
    pass
