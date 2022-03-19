import numpy as np

from model import *

from pre_process import *

import matplotlib.pyplot as plt

__image_directory = '../EMDS-5'
__image_extension = '.png'
__rotate_angels = [0, 90, 180, 270]
__aug_dump = '../Data/aug'
__original_source = '../EMDS-5/EMDS5-Original'
__gt_source = '../EMDS-5/EMDS5-GTM'
__gt_dump = '../Data/gt'

(trainX, trainY), (valX, valY), (testX, testY) = train_test_validation_split(__aug_dump, __gt_dump, 100)


plt.imshow(trainX[5])
plt.show()
print(trainX.shape)
plt.imshow(trainY[5])
plt.show()
print(trainY.shape)

plt.hist(np.ndarray.flatten(trainY[0]))
plt.show()
