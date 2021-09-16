import tensorflow as tf
import numpy as np
import datetime
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import sklearn
import tkinter


class FilterDataset:

    def __init__(self, gt_label=0, num_of_corr=32, num_of_images=None, augmentation=None, inverse=False,
                 sample_weight=(1, 1, 1)):
        self.gt_label = gt_label
        self.num_of_corr = num_of_corr
        self.num_of_images = num_of_images
        self.augmentation = augmentation
        self.sample_weights = sample_weight
        self.inverse = inverse

    def prepare_data_from_array(self, train_images, train_labels, validation_images=None, validation_labels=None,
                                num_of_images_last=False):

        if not validation_images.any():
            train_images, train_labels, validation_images, validation_labels = train_test_split(train_images,
                                                                                                train_labels,
                                                                                                shuffle=True,
                                                                                                test_size=0.33,
                                                                                                random_state=42)

        if not (3 <= len(train_images.shape) <= 4 or 3 <= len(validation_images.shape) <= 4):
            raise ValueError('Train data shape should be 3 or 4 dimensional! Got {}-dimensional data array!'.format(
                                                                                                 len(train_images.shape)
            ))
        if num_of_images_last:
            validation_images = np.transpose(validation_images)
            train_images = np.transpose(train_images)

        if not self.num_of_images or self.num_of_images > train_images.shape[0]:
            self.num_of_images = train_images.shape[0]
            print('num_of_images set to {}!'.format(self.num_of_images))

        train_images, train_labels = self._balance_class(images=train_images, labels=train_labels)
        print('num_of_images set to {}!'.format(train_images.shape[-1]))
        train_weights = self._make_data_generator(images=train_images, labels=train_labels)

        self.augmentation = None
        validation_images, validation_labels = self._balance_class(validation_images, validation_labels)
        validation_weights = self._make_data_generator(images=validation_images,
                                                       labels=validation_labels)

        return train_weights, validation_weights, train_images.shape

    def prepare_data_from_directory(self, train_path, validation_path=None, train_labels_path=None,
                                    validation_labels_path=None, target_size=(100, 100),
                                    initial_filter_matrix=np.zeros(1)):
        if train_labels_path:
            train_image_names, train_labels, = self.get_data_from_csv(train_labels_path, train_path)
            if validation_labels_path:
                validation_image_names, validation_labels = self.get_data_from_csv(validation_labels_path,
                                                                                   validation_path)
        else:
            train_image_names, train_labels = self.get_data_from_directory(train_path)
            if validation_path:
                validation_image_names, validation_labels = self.get_data_from_directory(validation_path)

        if not validation_path:
            train_image_names, validation_image_names, train_labels, validation_labels = train_test_split(
                                                                                                    train_image_names,
                                                                                                    train_labels,
                                                                                                    shuffle=True,
                                                                                                    test_size=0.33,
                                                                                                    random_state=42
                                                                                                          )
        if not self.num_of_images or self.num_of_images > len(train_image_names):
            self.num_of_images = len(train_image_names)
            print('num_of_images set to {}!'.format(self.num_of_images))

        train_image_names, train_labels = self._balance_class(images=train_image_names, labels=train_labels)
        print(len(train_image_names))

        train_weights = self._make_data_generator_from_directory(images=train_image_names,
                                                                 labels=train_labels,
                                                                 target_size=target_size,
                                                                 initial_filter_matrix=initial_filter_matrix)

        self.augmentation = None
        validation_image_names, validation_labels = self._balance_class(images=validation_image_names,
                                                                        labels=validation_labels)
        validation_weights = self._make_data_generator_from_directory(images=validation_image_names,
                                                                      labels=validation_labels,
                                                                      target_size=target_size,
                                                                      initial_filter_matrix=initial_filter_matrix)

        return train_weights, validation_weights, len(train_image_names)

    @staticmethod
    def get_data_from_csv(labels_path, path):
        df = pd.read_csv(labels_path)
        file_names = df.iloc[:, 0].tolist()
        labels = df.iloc[:, 1].to_numpy()
        images_path = list(map(lambda x: path + '/' + x, file_names))
        return images_path, labels

    @staticmethod
    def get_data_from_directory(path):
        labels = []
        images_path = []
        for label, directory in enumerate(list(map(lambda x: path + '/' + x, os.listdir(path)))):
            images = list(map(lambda x: directory + '/' + x, os.listdir(directory)))
            images_path = images_path + images
            labels = labels + list(label * np.ones(len(images)))
        return images_path, np.array(labels)

    def _balance_class(self, images, labels, test_mode=False):
        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = np.expand_dims(images, 3)
            elif images.shape[3] > 1:
                images = np.mean(images, axis=3, keepdims=True)
        else:
            images = np.array(images)

        positive_images = images[labels == self.gt_label]
        positive_labels = labels[labels == self.gt_label]
        negative_images = images[labels != self.gt_label]
        negative_labels = labels[labels != self.gt_label]
        if self.num_of_images // 2 < len(positive_images):
            if isinstance(images[0], str):
                positive_images = positive_images[:self.num_of_images // 2]
                negative_images = negative_images[:self.num_of_images // 2]
            else:
                positive_images = positive_images[:self.num_of_images // 2, :, :, :]
                negative_images = negative_images[:self.num_of_images // 2, :, :, :]
            positive_labels = positive_labels[:self.num_of_images // 2]
            negative_labels = negative_labels[:self.num_of_images // 2]
        else:
            if isinstance(images[0], str):
                negative_images = negative_images[:positive_images.shape[0]]
            else:
                negative_images = negative_images[:positive_images.shape[0], :, :, :]
            negative_labels = negative_labels[:positive_images.shape[0]]

        if isinstance(images[0], str):
            images = np.hstack((positive_images, negative_images))
        else:
            images = np.vstack((positive_images, negative_images))
        labels = np.hstack((positive_labels, negative_labels))
        if not test_mode:
            if len(images) % self.num_of_corr:
                if isinstance(images[0], str):
                    images = images[:len(images) - len(images) % self.num_of_corr]
                else:
                    images = images[:images.shape[0] - images.shape[0] % self.num_of_corr, :, :, :]
                labels = labels[:len(images) - len(images) % self.num_of_corr]

        labels = labels == self.gt_label
        labels = labels.astype(int)
        return images, labels

    def _make_data_generator(self, images, labels):
        gt, sample_weight = self._make_gt_correlation(shape=images.shape, labels=labels)
        if self.augmentation:
            rotation_range, horizontal_flip, vertical_flip = self.augmentation
            datagen = ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=rotation_range,
                                         horizontal_flip=horizontal_flip,
                                         vertical_flip=vertical_flip)
        else:
            datagen = ImageDataGenerator(rescale=1 / 255.)

        data = datagen.flow(images, gt, sample_weight=sample_weight, seed=42, batch_size=self.num_of_corr)
        return data

    def _make_data_generator_from_directory(self, images, labels, target_size, initial_filter_matrix):
        gt, sample_weight = self._make_gt_correlation(shape=(len(images), target_size[0], target_size[1], 1),
                                                      labels=labels)
        datagen = CustomDataGen(images_path=images,
                                labels=gt,
                                batch_size=self.num_of_corr,
                                input_size=(target_size[0], target_size[1], 1),
                                sample_weight=sample_weight,
                                shuffle=True,
                                is_train=True,
                                inverse=self.inverse,
                                filter_matrix=initial_filter_matrix)
        return datagen

    def _make_gt_correlation(self, shape, labels):
        gt = np.zeros(shape, dtype='float32')
        class_weights = np.ones(shape, dtype='float32')*self.sample_weights[0]

        if shape[1] % 2:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 2
        else:
            x1, x2 = shape[1] // 2 - 3, shape[1] // 2 + 3

        gt[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1
        class_weights[np.where(labels != self.gt_label), :, :, :] = 1 * self.sample_weights[1]
        class_weights[np.where(labels != self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[2]
        class_weights[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[2]
        return gt, class_weights

    def _get_gt_correlation(self, shape, labels):
        gt = np.zeros(shape, dtype='float32')
        class_weights = np.ones(shape, dtype='float32') * self.sample_weights[0]

        if shape[1] % 2:
            x1, x2 = shape[1] // 2 - 1, shape[1] // 2 + 2
        else:
            x1, x2 = shape[1] // 2 - 3, shape[1] // 2 + 3
        # np.load('negative_correlation')
        gt[np.where(labels == self.gt_label), x1: x2, x1: x2, :] = np.load('gt_correlation')
        gt[np.where(labels != self.gt_label), :, :, :] = 0.1
        class_weights[np.where(labels != self.gt_label), :, :, :] = 1 * self.sample_weights[1]
        class_weights[np.where(labels == self.gt_label), x1:x2, x1:x2, :] = 1 * self.sample_weights[2]
        return gt, class_weights

    def make_test_data_from_array(self, images, labels):
        images, labels = self._balance_class(images=images, labels=labels, test_mode=True)
        return images, labels

    def make_test_data_from_directory(self, path, labels_path=None, target_size=(100, 100)):

        if labels_path:
            image_names, labels, = self.get_data_from_csv(labels_path, path)
        else:
            names, labels = self.get_data_from_directory(path)

        if not self.num_of_images or self.num_of_images > len(image_names):
            self.num_of_images = len(image_names)
            print('num_of_images set to {}!'.format(self.num_of_images))

        image_names, labels = self._balance_class(images=image_names, labels=labels)
        images = []
        for path in enumerate(image_names):
            images.append(self._get_image(path, target_size=target_size))
        print(len(image_names))

        return images, labels

    def _get_image(self, path, target_size):

        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(target_size[0],
                                                                                                 target_size[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)
        if self.inverse:
            image = 255 - image
        return image / 255.


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, images_path, labels, batch_size, sample_weight=np.zeros(1),
                 input_size=(256, 256, 1),  shuffle=True, is_train=True, inverse=False, filter_matrix=np.zeros(1)):
        self.images_path = images_path
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.number_of_samples = len(self.labels)
        self.sample_weights = sample_weight
        self.inverse = inverse
        if filter_matrix.any():
            if len(filter_matrix.shape) > 2:
                filter_matrix = np.mean(filter_matrix, axis=-1)
            if filter_matrix.shape[0] != self.input_size[0] or filter_matrix.shape[1] != self.input_size[1]:
                raise ValueError("The initial filter matrix size is incorrect! Find: {}, should be: {}".format(
                    filter_matrix.shape,
                    self.input_size
                ))
            filter_matrix = np.expand_dims(filter_matrix, axis=(0, 3))
            self.filter_matrix = np.repeat(filter_matrix, self.batch_size, axis=0)
        else:
            self.filter_matrix = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))

    def on_epoch_end(self):
        if self.is_train:
            shuffler = np.random.permutation(len(self.images_path))
            self.images_path = self.images_path[shuffler]
            self.labels = self.labels[shuffler]
            if self.sample_weights.any():
                self.sample_weights = self.sample_weights[shuffler]

    def __getitem__(self, index):
        batch_images_path = self.images_path[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]
        labels = batch_labels[:, batch_labels.shape[1] // 2, batch_labels.shape[2] // 2, 0]
        labels = tf.keras.utils.to_categorical(labels, num_classes=2)
        batch_sample_weights = self.sample_weights[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]
        batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))
        initial_filter_matrix = np.ones((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))
        for i, path in enumerate(batch_images_path):
            batch_images[i, :, :, :] = self._get_image(path)

        return [initial_filter_matrix, np.float32(batch_images)], [batch_labels, labels], batch_sample_weights

    def __len__(self):
        return self.number_of_samples // self.batch_size

    def _get_image(self, path):

        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(self.input_size[0],
                                                                                                 self.input_size[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)
        if self.inverse:
            image = 255 - image
        return image / 255.


class SetWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator):
        super(SetWeightsCallback, self).__init__()
        self.generator = generator

    def on_train_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)

    def on_test_batch_begin(self, batch, logs=None):
        weights, _, _ = self.generator[batch]
        weights = np.expand_dims(weights, 4)
        weights = np.transpose(weights, [3, 1, 2, 4, 0])
        self.model.get_layer('correlation').set_weights(weights)


def make_tensorboard():
    path = "logs/fit/"
    if os.path.exists(path):
        shutil.rmtree(path)
    log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard


def PrepareSVHS():

    train = loadmat(r'D:\MIFI\SCIENTIFIC WORK\DATASETS\SVHN dataset\train_32x32.mat')
    test = loadmat(r'D:\MIFI\SCIENTIFIC WORK\DATASETS\SVHN dataset\train_32x32.mat')
    train, labels_train = train['X'][:, :, :, :], train['y'][:]
    test, labels_test = test['X'][:, :, :, :], test['y'][:]
    train = np.transpose(train, axes=[3, 0, 1, 2])
    test = np.transpose(test, axes=[3, 0, 1, 2])
    train = train[labels_train[:, 0] == 5]
    print(train.shape)
    return train, test, labels_train, labels_test


def filter_classes(dataset, classes):
    """
    This function should filter the dataset by only retaining dataset elements whose
    label belongs to one of the integers in the classes list.
    The function should then return the filtered Dataset object.
    """
    dataset = dataset.filter(lambda images, labels: tf.reduce_any(tf.equal(labels, tf.constant(classes, dtype=tf.int64))))
    return dataset


def mse_loss(pred, target):
    loss = tf.math.reduce_mean(tf.square(target - pred), axis=3, keepdims=True)
    return loss


def mean_loss(loss):
    sum = tf.reduce_sum(loss, axis=(1, 2))
    return tf.reduce_mean(sum)


def learning_curves_plot(train_loss, train_accuracy):
    fig, axes = plt.subplots(1, 2, sharex=False, figsize=(18, 5))

    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_title('Loss vs epochs')
    axes[0].plot(train_loss)

    axes[1].set_title('Accuracy vs epochs')
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].plot(train_accuracy)
    plt.show()

#tensorboard --logdir logs/fit
