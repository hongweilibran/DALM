from scipy.io import loadmat
import numpy as np
from PIL import Image
import os
import random
from imgaug import augmenters as iaa

def load_data(train_list, val_list, augment=True):
    augment_size = 150  #define how many times the augmented dataset comparing to the original images.
    ## one-hot conversion
    def convert_to_onehot(label, numClass):
        one_hot = np.zeros((1, label.shape[0], label.shape[1], numClass), dtype=np.float32)
        for i in range(numClass):
            one_hot[0, :, :, i][label == i] = 1
        return one_hot

    ## paramters of the image size
    #     IMG_WIDTH = 96
    #     IMG_HEIGHT = 96
    #     IMG_CHANNELS = 254
    NUM_class = 5
    data_path = 'Dataset/'

    X_train = []  # training data
    y_train = []  # training label
    X_val = []  # validation data
    y_val = []  # validation label
    for nn in train_list:
        img = loadmat(os.path.join(data_path, 'data', 'T' + nn + '.mat'))['tr']
        label = np.asarray(Image.open(os.path.join(data_path, 'label', 'L' + nn + '.png')))
        one_hot_label = np.squeeze(convert_to_onehot(label, NUM_class))
        X_train.append(img)
        y_train.append(one_hot_label)
        if augment:
            for aa in range(augment_size):
                seed = random.randint(1, 123456789)
                #image augmentation for changing the colors
                aug_color = iaa.Sequential([iaa.color.ChangeColorTemperature(5000, from_colorspace='RGB')], random_order=True)
                aug_rest = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-15, 15),
                                          order=0, mode="constant", cval=0),
                                          iaa.Rot90([1, 2, 3])], seed=seed, random_order=True)  
                
                joint_array = np.concatenate((img[np.newaxis, ...], label[np.newaxis, ..., np.newaxis]), axis = -1)
                joint_new = aug_rest(images = joint_array)
                augmted_images = joint_new[0, :, :, :-1]
                augmted_masks = joint_new[0, :, :, -1]
                one_hot_label = np.squeeze(convert_to_onehot(augmted_masks[ :, :], NUM_class))
                X_train.append(np.float32(augmted_images))
                y_train.append(one_hot_label)
                
                seed = random.randint(1, 123456789)
                aug_rest = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-15, 15), 
                                          order=0, mode="constant", cval=0), iaa.Fliplr(1),
                                          iaa.Rot90([1, 2, 3])], seed=seed, random_order=True)
                
                augmted_mic = aug_color(images = np.uint8(img[np.newaxis, ..., 0:3]*255))
                augmted_hs = aug_rest(images = img[np.newaxis, ..., 3:])
                
                augmted_mic = (np.float32(augmted_mic)-np.min(augmted_mic))/(np.max(augmted_mic)-np.min(augmted_mic))
                augmted_masks = aug_rest(images = label[np.newaxis, ..., np.newaxis])
                one_hot_label_ = np.squeeze(convert_to_onehot(augmted_masks[0, :, :, 0], NUM_class))
                augmted_images_ = np.concatenate((np.float32(augmted_mic), augmted_hs), axis = -1)
                X_train.append(np.float32(augmted_images_[0, ...]))
                y_train.append(one_hot_label_)

    img = loadmat(os.path.join(data_path, 'data', 'T' + val_list[0] + '.mat'))['tr']
    label = np.asarray(Image.open(os.path.join(data_path, 'label', 'L' + val_list[0] + '.png')))
    one_hot_label = np.squeeze(convert_to_onehot(label, NUM_class))
    X_val.append(img)
    y_val.append(one_hot_label)
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    return X_train, y_train, X_val, y_val


def main():
    train_list = ['7', '8_2', '9_4', '14', '20_2', '20', '23', '30'] #define the list of training samples
    val_list = ['8', '10_2', '10', '22_1', '31'] #define the list of validation samples
    load_data(train_list, val_list, augment=True)


if __name__ == "__main__":
    main()
