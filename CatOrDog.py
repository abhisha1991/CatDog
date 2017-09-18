import cv2                 # for resizing images
import numpy as np
import os
from random import shuffle
from tqdm import tqdm      # for visualization
import datetime
import matplotlib.pyplot as plt  # for visualization

# tflearn imports for model learning
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = CURR_DIR + '\\dogs_vs_cats\\training_set'
TEST_DIR = CURR_DIR + '\\dogs_vs_cats\\test_set'
IMG_SIZE = 64 # choosing squared pixel-ed matrix only [BW images used for now]
LR = 1e-3
EPOCH = 10
ACTIVATION_INTERMEDIATE = 'relu'
ACTIVATION_FINAL = 'softmax'
OPTIMIZER = ['adam', 'momentum', 'sgd', 'rmsprop']
LOG_DIR = CURR_DIR + "\\LOG"
VALIDATION_SIZE = 1000  # size of training set cut off for validation set
MODEL_NAME = 'dogsvscats-{}.model'.format(datetime.datetime.today().strftime('%Y-%m-%d'))


def label_img(img):
    word_label = img.split('.')[-3]
    # Image of the format dog.1.png
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# 6 layer conv net with max pooling
convnet = conv_2d(convnet, 128, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 128, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 128, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 128, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 64, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 32, 4, activation=ACTIVATION_INTERMEDIATE)
convnet = max_pool_2d(convnet, 4)

convnet = fully_connected(convnet, 1024, activation=ACTIVATION_INTERMEDIATE)
convnet = dropout(convnet, 0.7)

convnet = fully_connected(convnet, 2, activation=ACTIVATION_FINAL)
convnet = regression(convnet, optimizer=OPTIMIZER[0], learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir=LOG_DIR.replace('\\', '/'))


train = train_data[:-VALIDATION_SIZE]
validation = train_data[-VALIDATION_SIZE:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

validation_X = np.array([i[0] for i in validation]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
validation_y = [i[1] for i in validation]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCH, validation_set=({'input': validation_X}, {'targets': validation_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)

test_data = process_test_data()
test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:25]):
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]

    # make a grid of m x n to accommodate 1st x examples from training set
    subplot = fig.add_subplot(5, 5, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    # print the data in the sub plot grid
    subplot.imshow(orig, cmap='gray')
    plt.title(str_label)
    plt.tight_layout()
    subplot.set_facecolor('yellow')
    subplot.axes.get_xaxis().set_visible(False)
    subplot.axes.get_yaxis().set_visible(False)

plt.interactive(False)
# For viewing Tensor board dashboard via cmd, do - 'tensorboard --logdir=foo:C:\users\test\...\log'
# This opens tensor board on localhost:6006
plt.savefig(LOG_DIR +'\\catdog.png' )

