import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from PIL import Image as I
from PIL import ImageOps
from sklearn.model_selection import train_test_split

__image_directory = '../EMDS-5'
__image_extension = '.png'
__rotate_angels = [0, 90, 180, 270]
__aug_dump = '../Data/aug'
__original_source = '../EMDS-5/EMDS5-Original'
__gt_source = '../EMDS-5/EMDS5-GTM'
__gt_dump = '../Data/gt'


def __process_and_save(source_dir, dump_dir, file_name, extension, rgb=True):
    img = I.open(os.path.join(source_dir, file_name + '.' + extension))
    if rgb:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    img = img.resize((256, 256), I.LANCZOS)

    for a in __rotate_angels:
        rotated_image = img.rotate(a)
        rotated_image.save(os.path.join(dump_dir, file_name + '_r' + str(a) + '.' + extension))
        flipped_image = ImageOps.mirror(rotated_image)
        flipped_image.save(os.path.join(dump_dir, file_name + '_m' + str(a) + '.' + extension))


def __augment(original_image_dir, aug_dump_dir, gt_source_dir, gt_dump_dir, num_images=-1):

    converted = 0

    for path, _, files in os.walk(original_image_dir):
        for file in files:
            print('\r', file, end='')
            if __image_extension in file:

                file_name = file.split('.')[0]
                extension = file.split('.')[-1]

                __process_and_save(original_image_dir, aug_dump_dir, file_name, extension)
                __process_and_save(gt_source_dir, gt_dump_dir, file_name + '-GTM', extension, False)

                converted += 1
                if converted > num_images > 0:
                    return


def convert(arr, binarize=False):
    if binarize:
        arr[arr <= 128] = 0
        arr[arr > 128] = 1
        arr = arr.astype(np.float32)

    if len(arr[0].shape) == 3:
        return arr

    return np.expand_dims(arr, axis=-1)


# noinspection PyPep8Naming
def train_test_validation_split(img_dir, label_dir, limit=-1):

    X = []
    Y = []
    cnt = 0

    for path, _, files in os.walk(img_dir):
        if cnt > limit > 0:
            break
        for file in files:
            if __image_extension in file:

                if '_r' in file:
                    label_file = file.split("_r")[0] + '-GTM_r' + file.split("_r")[1]
                else:
                    label_file = file.split("_m")[0] + '-GTM_m' + file.split("_m")[1]

                img = I.open(os.path.join(img_dir, file))
                X.append(np.array(img))

                img = I.open(os.path.join(label_dir, label_file))
                Y.append(np.array(img))
                cnt += 1

                if cnt > limit > 0:
                    break

    trainX, X, trainY, Y = train_test_split(X, Y, train_size=.25, random_state=3)
    valX, testX, valY, testY = train_test_split(X, Y, train_size=1/3, random_state=3)

    trainX = convert(np.array(trainX))
    trainY = convert(np.array(trainY), True)
    valX = convert(np.array(valX))
    valY = convert(np.array(valY), True)
    testX = convert(np.array(testX))
    testY = convert(np.array(testY), True)

    return (trainX, trainY), (valX, valY), (testX, testY)


if __name__ == "__main__":
    __augment(__original_source, __aug_dump, __gt_source, __gt_dump)

    print("\nAugment complete!")

    # (trainX, trainY), (valX, valY), (testX, testY) = train_test_validation_split(__aug_dump, __gt_dump)



