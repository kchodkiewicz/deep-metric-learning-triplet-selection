# from pprint import pprint
import cv2
import os
# import time
from tqdm import tqdm
# from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
import pickle
from pprint import pprint

import numpy as np
# from numpy import random as rng
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation, Dropout
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras import backend as K


def load_data(dir_path):
    X = []
    y = []
    lang_dict = {}
    data_dict = {}
    classNo = 0
    for alphabet in tqdm(sorted(os.listdir(dir_path))):
        lang_dict[alphabet] = [classNo,None]
        alpha_path = os.path.join(dir_path,alphabet)
        for letter in sorted(os.listdir(alpha_path)):
            cat_images = []
            for img in sorted(os.listdir(os.path.join(alpha_path,letter))):
                img_path = os.path.join(alpha_path,letter,img)
                cat_images.append(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
                y.append(classNo)
            X.append(cat_images)
            lang_dict[alphabet][1] = classNo
            data_dict[classNo] = cat_images
            classNo += 1
        break
    X = np.array(X)
    y = np.array(y)
    return data_dict



image_train_path = r'S:\studia\uni\praca_magisterska\deep-metric-learning-triplet-selection\code\deep-metric-learning\data\omniglot\images_background\images_background'
image_eval_path = r'S:\studia\uni\praca_magisterska\deep-metric-learning-triplet-selection\code\deep-metric-learning\data\omniglot\images_evaluation\images_evaluation'

train_dict = load_data(image_train_path)
val_dict = load_data(image_eval_path)

# print(train_dict, end='\n\n')
# print(type(train_dict), end='\n\n')

with open('s:/studia/uni/praca_magisterska/deep-metric-learning-triplet-selection/code/deep-metric-learning/data/omniglot_train_dict.pkl', 'wb') as f:
    pickle.dump(train_dict, f)

with open('s:/studia/uni/praca_magisterska/deep-metric-learning-triplet-selection/code/deep-metric-learning/data/omniglot_val_dict.pkl', 'wb') as f:
    pickle.dump(val_dict, f)

        
with open('omniglot_train_dict.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('omniglot_val_dict.pkl', 'rb') as f:
    val_data = pickle.load(f)

# print(train_data, end='\n\n')
# print(type(train_data), end='\n\n')


# import matplotlib.pyplot as plt
# import numpy as np

# def plot_image_grid(images, ncols=None, cmap='gray'):
#     '''Plot a grid of images'''
#     if not ncols:
#         factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
#         ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1

#     nrows = int(len(images) / ncols) + int(len(images) % ncols)
#     imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
#     axes = axes.flatten()[:len(imgs)]

#     for img, ax in zip(imgs, axes.flatten()): 
#         if np.any(img):
#             if len(img.shape) > 2 and img.shape[2] == 1:
#                 img = img.squeeze()
#             ax.imshow(img, cmap=cmap)
#             ax.grid(False)
#             # ax.set_title(labels[i])
#             ax.axis("off")
#     fig.show()


# with open('omniglot_train_dict.pkl', 'rb') as f:
#   train_data = pickle.load(f)

# printable_data = []

# for num in range(len(train_data[0])):
#   for row in zip(*[train_data[key][num] for key in train_data.keys()]):
#     for item in row:
#       printable_data.append(item)

# plot_image_grid(printable_data, ncols=len(train_data.keys()))
