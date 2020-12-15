import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D,  UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.utils import plot_model, to_categorical
from keras import backend as keras
# from data import *
import cv2
from keras import utils
import glob
import random
import keras.backend as K
from skimage import io
from shutil import copyfile
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Reshape,Conv2DTranspose
from keras.layers import Conv2D, Flatten, Lambda
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageChops
# from sklearn.neighbors import KernelDensity
from keras.utils import plot_model

class img_generator:

    def __init__(self, image_list, label_list, bitch_size=2):
        self.image_list = image_list
        self.label_list = label_list
        self.bitch_size = bitch_size

    def __iter__(self):

        images, labels = [], []
        while True:

            index = random.randint(0, len(self.image_list)-1)
            image = Image.open(self.image_list[index])
            image = image.resize((256,256))
            image = np.asarray(image)
            image = image.reshape(256,256,1)
            masks = Image.open(self.label_list[index])
            masks = masks.resize((256,256))
            masks = np.asarray(masks)
            masks = masks.reshape(256, 256, 1)  #key
            masks = utils.to_categorical(masks, 2)
            image = image.astype('float32')
            masks = masks.astype('float32')
            # data preprocess
            mean = np.mean(image)
            std = np.std(image)
            image -= mean
            image /= std
            images.append(image)
            labels.append(masks)
            if len(images) >= self.bitch_size:
                images = np.array(images)
                labels = np.array(labels)
                yield (images, labels)
                images, labels = [], []

#smooth = 1.
dropout_rate = 0.4
act = "relu"

#model
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', border_mode='same')(input_tensor)
    #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', border_mode='same')(x)
    #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)
    return x

def Nest_Net(img_rows, img_cols, color_type=1, num_class=2, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    img_input = Input(shape = (img_rows,img_cols,color_type))
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2) ,strides= (2,2) ,name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', border_mode='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12',axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', border_mode='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22',axis=3)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', border_mode='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13',axis=3)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', border_mode='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32',axis=3)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', border_mode='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23',axis=3)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', border_mode='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14',axis=3)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', border_mode='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42',axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', border_mode='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33',axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', border_mode='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24',axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', border_mode='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15',axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', border_mode='same')(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', border_mode='same')(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', border_mode='same')(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', border_mode='same')(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model


log_dir = 'logsPretrain/'


model = Nest_Net(256,256)
model.summary()


es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
#model.compile(optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file="model.png", show_shapes=True)

model_checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    period=3)

# 学习率下降的方式，val_loss 2次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=6,
    verbose=1
)

# tr_image_list1 = glob.glob(r'***.tif')
# tr_label_list1 = glob.glob(r'***.tif')
# va_image_list1 = glob.glob(r'***.tif')
# va_label_list1 = glob.glob(r'***.tif')
# te_image_list1 = glob.glob(r'***.tif')
#
#
# train_generator = img_generator(image_list=tr_image_list1, label_list=tr_label_list1, bitch_size=2)
# val_generator = img_generator(image_list=va_image_list1, label_list=va_label_list1, bitch_size=2)
# #此版本：分训练验证和测验+对数据进行预处理+均值方差代替除以255（label没有进行预处理可不可以）
#
# model.fit_generator(generator=train_generator.__iter__(), steps_per_epoch=300, epochs=50, verbose=1,
#                      callbacks=[es,model_checkpoint], validation_data=val_generator.__iter__(),
#                      validation_steps=100)
#
# model.save_weights(log_dir + 'last.h5')

# score = model.evaluate(test_x, test_y, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# model.fit(x=train_data, y=train_label,
#           batch_size=10, epochs=5,
#           validation_data=(val_data, val_label))

# for i , image_file in enumerate(te_image_list1):
#     #image1 = io.imread(image_file)
#     image1 = Image.open(image_file)
#     image1 = image1.resize((256,256))
#     image1 = np.asarray(image1)
#     image1 = image1.reshape(256,256,1)
#     image1 = image1.astype('float32')
#     mean = np.mean(image1)
#     std = np.std(image1)
#     image1 -= mean
#     image1 /= std
#     #image1 = image1/255.0
#     #image1 = image1.astype('float32')
#     test = np.zeros(shape = (1,256,256,1), dtype = 'float32')
#     test[0,:,:,:] = image1
#     results = model.predict(test)
#     re = results[0,:,:,:]
#     mask = np.argmax(re, axis=2)
#     print(np.max(mask))
#     io.imsave('./result1/' + str(i + 1).zfill(6) + '.png', (mask * 255).astype('uint8'))

# for i in range(len(test_idx)):
#     test = test_data[i:i+1, :, :]
#     results = model.predict(test)
#     re = results[0, :, :]
#     mask = np.argmax(re, axis=2)
#     print(np.max(mask))
#     io.imsave('./result/' + str(i+1).zfill(5) + '.png', (mask * 255).astype('uint8'))
