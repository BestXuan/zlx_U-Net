import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import utils
import glob
import random
from skimage import io
from PIL import Image
import cv2
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


#这个是图像的生成器，因为内存不够，所以只能一张一张的放图片进去。
class img_generator:
    def __init__(self, image_list, label_list, bitch_size = 1):
        self.image_list = image_list
        self.label_list = label_list
        self.bitch_size = bitch_size

    def __iter__(self):
        images, labels = [], []
        while True:
            index = random.randint(0, len(self.image_list) - 1)
            image = Image.open(self.image_list[index])
            image = image.resize((512, 512))
            image = np.asarray(image)
            image = image.reshape(512,512,1)
            image = image / 255.0
            masks = Image.imread(self.label_list[index])
            masks = np.resize(masks,(512,512))
            masks = np.asarray(masks)
            masks = masks.reshape(512, 512, 1)
            # masks = masks / 255 #在生物数据上需要，但是在材料数据上不需要(因为它本身就是1和0的)
            # print(masks)
            masks = utils.to_categorical(masks, 2)
            image = image.astype('float32')
            masks = masks.astype('float32')
            # 下面是数据的标准化。
            # mean = np.mean(image)
            # std = np.std(image)
            # image -= mean
            # image /= std
            images.append(image)
            labels.append(masks)
            if len(images) >= self.bitch_size:
                images = np.array(images)
                labels = np.array(labels)
                yield (images, labels)
                images, labels = [], []


if __name__ == '__main__':
    #这里是读数据
    image_list = glob.glob(r'***.tif')
    label_list = glob.glob(r'***.tif')
    test_list = glob.glob(r'***.tif')

    train_generator = img_generator(image_list=image_list, label_list=label_list, bitch_size=1)
    val_generator = img_generator(image_list=image_list, label_list=label_list, bitch_size=1)

    #这个是搭建模型
    inputs = Input((256, 256, 1))
    conv1 = Conv2D(64, 3,  activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3,  activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3,  activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3,  activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3,  activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3,  activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3,  activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3,  activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3,  activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3,  activation='relu', padding='same')(conv5)

    upmid = concatenate([Convolution2D(512, 2,  padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4],axis=3)
    convmid = Conv2D(512, 3,  activation='relu', padding='same')(upmid)
    convmid = Conv2D(512, 3,  activation='relu', padding='same')(convmid)

    up6 = concatenate( [Convolution2D(256, 2,  activation='relu', padding='same')(UpSampling2D(size=(2, 2))(convmid)), conv3],axis=3)
    conv6 = Conv2D(256, 3,  activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3,  activation='relu', padding='same')(conv6)

    up7 = concatenate([Convolution2D(128, 2,  activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv2],axis=3)
    conv7 = Conv2D(128, 3,  activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3,  activation='relu', padding='same')(conv7)

    up8 = concatenate([Convolution2D(64, 2,  activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv1],axis=3)
    conv8 = Conv2D(64, 3,  activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3,  activation='relu', padding='same')(conv8)

    conv9 = Conv2D(2, 1,  activation='sigmoid')(conv8)

    model = Model(input=inputs, output=conv9)
    model.summary()

    model.compile(optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint('./unet2_{epoch:04d}_{val_loss:.05f}.hdf5', monitor='val_loss', verbose=1,save_best_only=True)  # 保存权重路径
    print('Fitting model...')

    model.fit_generator(generator=train_generator.__iter__(), steps_per_epoch=30, epochs=100)

    #现在生成分割后的图片
    for i, image_file in enumerate(test_list):
        image = Image.imread(image_file)
        image = image.resize(512, 512)
        image = np.asarray(image)
        image = image.reshape(512, 512, 1)
        image = image / 255.0
        image = image.astype('float32')
        test = np.zeros(shape=(1, 512, 512, 1), dtype='float32')
        test[0, :, :, :] = image
        results = model.predict(test)
        re = results[0, :, :, :]
        mask = np.argmax(re, axis=2)
        # print(np.max(mask))
        io.imsave('./result/' + str(i + 1).zfill(5) + '.png', (mask * 255).astype('uint8'))
        print("finish the "+str(i))