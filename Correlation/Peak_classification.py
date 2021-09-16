from typing import Tuple, Optional
from tensorflow import keras
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Add, Activation
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from Correlation_utils import Correlator
import pandas as pd
import os
import cv2 as cv
import tensorflow as tf


class CustomResNet18:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, alpha: float,
                 regularization: Optional[float], activation_type: str, input_name: str):
        """
        Custom implementation of the stripped-down ResNet18 for classification task.
        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param alpha: network expansion factor, determines the number of filters in each layer.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param activation_type: type of activation function. See cv_utils.Activation.
        :param input_name: name of the input tensor.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_type = activation_type
        self.input_name = input_name

        self.init_filters = int(16 * alpha)
        self.ker_reg = None if regularization is None else keras.regularizers.l2(regularization)

    def build(self) -> keras.models.Model:
        """
        Building CNN model for classification task.
        :return: keras.model.Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)
        x = BatchNormalization()(inputs)
        x = Conv2D(self.init_filters, (3, 3), strides=2, use_bias=False, kernel_regularizer=self.ker_reg)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation_type)(x)

        x = self.res_block(x, self.init_filters * 2, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2)

        x = self.res_block(x, self.init_filters * 2 ** 2, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2 ** 2)

        x = self.res_block(x, self.init_filters * 2 ** 3, 2)
        for _ in range(2):
            x = self.res_block(x, self.init_filters * 2 ** 3)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return keras.models.Model(inputs=inputs, outputs=x)

    def res_block(self, x: Tensor, filters: int, stride: int = 1) -> Tensor:
        """
        Residual block. If stride == 1, then there are no any transformations in one of the branches.
        If stride > 1, then there are convolution with 1x1 filters in one of the branches.
        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :param stride: convolution stride.
        :return: output tensor.
        """
        conv_kwargs = {'use_bias': False, 'padding': 'same', 'kernel_regularizer': self.ker_reg}
        x1 = Conv2D(filters, (3, 3), strides=stride, **conv_kwargs)(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(self.activation_type)(x1)
        x1 = Conv2D(filters, (3, 3), **conv_kwargs)(x1)
        if stride == 1:
            x2 = x
        else:
            x2 = Conv2D(filters, (1, 1), strides=stride, **conv_kwargs)(x)
        x_out = Add()([x1, x2])
        x_out = BatchNormalization()(x_out)
        x_out = Activation(self.activation_type)(x_out)
        return x_out


class CorrelationGenerator:

    def __init__(self, images_path, num_of_correlations, labels_path=None, input_size=(256, 256), inverse=False,
                 batch_size=32):
        self.input_size = input_size
        self.number_of_samples = num_of_correlations
        self.images_names, self.labels = self.get_data(labels_path, images_path)
        self.inverse = inverse
        self.batch_size = batch_size

    def get_generator(self):
        correlations = np.zeros((self.number_of_samples, self.input_size[0], self.input_size[1]))
        correlation_labels = np.zeros(self.number_of_samples)
        for i in range(self.number_of_samples):
            if i % 2:
                path = np.random.choice(self.images_names, 1, replace=False)[0]
                correlations[i, :, :] = self._get_correlation((path, path))
                correlation_labels[i] = 1
            else:
                x = np.random.choice(len(self.images_names), 1, replace=False)[0]
                path1 = self.images_names[x]
                label1 = self.labels[x]
                images_name = [image for label, image in zip(self.labels, self.images_names) if label != label1]
                path2 = np.random.choice(images_name, 1, replace=False)[0]
                correlations[i, :, :] = self._get_correlation((path1, path2))
                correlation_labels[i] = 0
        
        correlations = self.min_max_scaler(np.expand_dims(correlations, axis=-1))
        correlation_labels = tf.keras.utils.to_categorical(correlation_labels, num_classes=2)
        np.save('correlations', correlations)
        np.save('correlation_labels', correlation_labels)
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, vertical_flip=True).flow(
            x=correlations,
            y=correlation_labels,
            batch_size=self.batch_size
        )
        return datagen

    def _get_correlation(self, path):
        c = Correlator(image=self._get_image(path[0]), cf=self._get_image(path[1]))
        return c.fourier_correlation()

    def _get_image(self, path):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (self.input_size[0], self.input_size[1]), interpolation=cv.INTER_AREA)
        if self.inverse:
            image = 255 - image
        return image / 255.

    def get_data(self, labels_path, images_path):
        if labels_path:
            return self.get_data_from_csv(labels_path, images_path)
        else:
            return self.get_data_from_directory(images_path)

    @staticmethod
    def get_data_from_csv(labels_path, path):
        df = pd.read_csv(labels_path)
        file_names = df.iloc[:, 0].tolist()
        label = df.iloc[:, 1].to_numpy()
        images_path = list(map(lambda x: path + '/' + x, file_names))
        return images_path, list(label)

    @staticmethod
    def get_data_from_directory(path):
        label = []
        images_path = []
        for x, directory in enumerate(list(map(lambda x: path + '/' + x, os.listdir(path)))):
            image = list(map(lambda x: directory + '/' + x, os.listdir(directory)))
            images_path = images_path + image
            label = label + list(x * np.ones(len(image)))
        return images_path, label

    @staticmethod
    def min_max_scaler(x):
        x = (x - np.min(x, axis=(1, 2), keepdims=True)) / (
                np.max(x, axis=(1, 2), keepdims=True) - np.min(x, axis=(1, 2), keepdims=True)
                + 1e-7
        ) + + 1e-7
        return np.abs(x)


# images = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
# labels = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
# data = CorrelationGenerator(images_path=images,
#                             num_of_correlations=32,
#                             labels_path=labels,
#                             input_size=(48, 48)).get_generator()
# x, y = data[0]
# print(x.max())
tf.keras.backend.set_learning_phase(0)
model = CustomResNet18(input_shape=(48, 48, 1), num_classes=2, alpha=1.,
                       regularization=0.0005, activation_type='relu', input_name='cpr').build()
model.summary()

