import numpy as np
import sklearn
import tkinter
import tensorflow as tf
import os
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Any


class CorrelationFilterDataset(ABC):
    def __init__(self, pos_class_label: int, batch_size: int = 32, num_of_images=None, inverse: bool = False) -> None:
        """
        Correlation filters dataset base class.
        :param pos_class_label: positive class label.
        :param inverse: parameter indicating whether to inverse image during preprocessing or not.
        :param num_of_images: number of images for the training set.
        :param batch_size: number of images in batch.
        """
        self.gt_label = pos_class_label
        self.num_of_images = num_of_images
        self.inverse = inverse
        self.batch_size = batch_size

    @abstractmethod
    def prepare_data_from_array(self, *args, **kwargs):
        pass

    @abstractmethod
    def prepare_data_from_directory(self, *args, **kwargs):
        pass

    def make_test_data_from_array(self, images: np.array, labels: np.array) -> Any:
        """
        Method for creating image data generator form numpy arrays.
        :parameter images: numpy array containing images.
        :parameter labels: numpy array containing labels.
        :returns image data generator.
        """
        images, labels = self._balance_class(images=images, labels=labels)
        return tf.keras.preprocessing.image.ImageDataGenerator().flow(
            np.array(images), labels, batch_size=self.batch_size
        )

    def make_test_data_from_directory(
            self, path: str, labels_path: str = None, target_size: Tuple[int, int] = (100, 100)
    ):
        """
        Method for creating image data generator form image data in folders.
        :parameter path: numpy array containing images.
        :parameter labels_path: path to csv data file with labels annotations.
        :parameter target_size: size to which all images will be resized.
        :returns image data generator.
        """
        if labels_path:
            image_names, labels, = self.get_data_from_csv(labels_path, path)
        else:
            image_names, labels = self.get_data_from_directory(path)

        if not self.num_of_images or self.num_of_images > len(image_names):
            self.num_of_images = len(image_names)

        image_names, labels = self._balance_class(images=image_names, labels=labels)

        images = list()
        for path in enumerate(image_names):
            images.append(self._get_image(path[1], target_size=target_size))

        print('num_of_images set to {}!'.format(np.shape(images)[0]))

        return tf.keras.preprocessing.image.ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True
        ).flow(np.array(images), labels, batch_size=self.batch_size)

    def _balance_class(self, images, labels, batch_round=False):
        """
        Method for balancing images by classes.
        :parameter images: numpy array containing images.
        :parameter labels: path to csv data file with labels annotations.
        :parameter batch_round: parameter indicating whether to drop last batch if it is not full or not.
        :returns image data generator.
        """
        if isinstance(images, list):
            images = np.array(images)
        else:
            if len(images.shape) == 3:
                images = np.expand_dims(images, 3)
            elif images.shape[3] > 1:
                images = np.mean(images, axis=3, keepdims=True)

        positive_images = images[labels == self.gt_label]
        positive_labels = labels[labels == self.gt_label]
        negative_images = images[labels != self.gt_label]
        negative_labels = labels[labels != self.gt_label]
        if self.num_of_images // 2 < len(positive_images):
            if isinstance(images[0], str):
                positive_images = np.random.choice(positive_images, self.num_of_images // 2)
                negative_images = np.random.choice(negative_images, self.num_of_images // 2)
            else:
                positive_images = positive_images[:self.num_of_images // 2, :, :, :]
                negative_images = negative_images[:self.num_of_images // 2, :, :, :]
            positive_labels = positive_labels[:self.num_of_images // 2]
            negative_labels = negative_labels[:self.num_of_images // 2]
        else:
            if isinstance(images[0], str):
                negative_images = np.random.choice(negative_images, positive_images.shape[0])
            else:
                negative_images = negative_images[:positive_images.shape[0], :, :, :]
            negative_labels = negative_labels[:positive_images.shape[0]]

        if isinstance(images[0], str):
            images = np.hstack((positive_images, negative_images))
        else:
            images = np.vstack((positive_images, negative_images))
        labels = np.hstack((positive_labels, negative_labels))
        if batch_round:
            if len(images) % self.batch_size:
                if isinstance(images[0], str):
                    images = images[:len(images) - len(images) % self.batch_size]
                else:
                    images = images[:images.shape[0] - images.shape[0] % self.batch_size, :, :, :]
                labels = labels[:len(images) - len(images) % self.batch_size]

        labels = labels == self.gt_label
        labels = labels.astype(int)
        return images, labels

    def _get_image(self, path: str, target_size: Tuple[int, int]) -> np.array:
        """
        Method for image loading.
        :parameter path: path to current image.
        :parameter target_size: size to which all images will be resized.
        :returns numpy array containing image.
        """
        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(target_size[0],
                                                                                                 target_size[1]))
        image = tf.keras.preprocessing.image.img_to_array(image)
        if self.inverse:
            image = 255 - image
        return image / 255.

    @staticmethod
    def standard_scaler(x: np.array) -> tf.Tensor:
        """
        Method for image loading.
        :parameter x: image array.
        :returns scaled image.
        """
        x = tf.squeeze(x, axis=-1) if len(x.shape) > 3 else x
        # x = tf.math.log(x + 1.)
        scaled_x = (x - tf.reduce_mean(x, axis=(-2, -1), keepdims=True)) / tf.math.reduce_std(
            x, axis=(-2, -1), keepdims=True
        )
        return scaled_x

    @staticmethod
    def get_data_from_csv(labels_path: str, path: str) -> Tuple[list, np.array]:
        """
        Method for getting images paths and labels from csv file.
        :parameter labels_path: path to labels annotations path.
        :parameter path: path to folder containing images.
        :returns images paths and labels.
        """
        df = pd.read_csv(labels_path)
        file_names = df.iloc[:, 0].tolist()
        labels = df.iloc[:, 1].to_numpy()
        images_path = list(map(lambda x: path + '/' + x, file_names))
        return images_path, labels

    @staticmethod
    def get_data_from_directory(path: str) -> Tuple[list, np.array]:
        """
        Method for getting images paths and labels from csv file.
        :parameter path: path to folder containing images.
        :returns images paths and labels.
        """
        labels = []
        images_path = []
        for label, directory in enumerate(list(map(lambda x: path + '/' + x, os.listdir(path)))):
            images = list(map(lambda x: directory + '/' + x, os.listdir(directory)))
            images_path = images_path + images
            labels = labels + list(label * np.ones(len(images)))
        return images_path, np.array(labels)


class CFDataset(CorrelationFilterDataset):
    """
    Classic correlation filters dataset class.
    """
    def prepare_data_from_array(self, train_images: np.array, train_labels: list, num_of_images_last=False) -> np.array:

        if not 3 <= len(train_images.shape) <= 4:
            raise ValueError('Train data shape should be 3 or 4 dimensional! Got {}-dimensional data array!'.format(
                                                                                                 len(train_images.shape)
            ))
        if num_of_images_last:
            train_images = np.transpose(train_images)

        if self.gt_label not in train_labels:
            raise ValueError(
                'Incorrect gt_label {}! Classes available: {}'.format(self.gt_label, np.unique(train_labels))
            )

        train_images = self._get_one_class(images=train_images, labels=train_labels)

        print('num_of_images set to {}!'.format(train_images.shape[-1]))

        if len(np.shape(train_images)) > 3:
            train_images = np.squeeze(np.array(train_images), axis=-1)

        return train_images

    def prepare_data_from_directory(
            self, train_path: str, train_labels_path: str = None, target_size: Tuple[int, int] = (100, 100)
    ) -> np.array:
        if train_labels_path:
            image_names, labels, = self.get_data_from_csv(train_labels_path, train_path)
        else:
            image_names, labels = self.get_data_from_directory(train_path)

        if self.gt_label not in labels:
            raise ValueError('Incorrect gt_label {}! Classes available: {}'.format(self.gt_label,
                                                                                   np.unique(labels)))

        image_names = self._get_one_class(images=image_names, labels=labels)
        images = []
        for path in enumerate(image_names):
            images.append(self._get_image(path[1], target_size=target_size))

        images = np.squeeze(np.array(images), axis=-1) if len(np.shape(images)) > 3 else images

        images = self.standard_scaler(np.array(images)).numpy()
        return images

    def _get_one_class(self, images, labels):
        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = np.expand_dims(images, 3)
            elif images.shape[3] > 1:
                images = np.mean(images, axis=3, keepdims=True)
        else:
            images = np.array(images)

        images = images[labels == self.gt_label]

        if self.num_of_images < len(images):
            if isinstance(images[0], str):
                images = images[:self.num_of_images]
            else:
                images = images[:self.num_of_images, :, :, :]

        return images


class MMCFDataset(CorrelationFilterDataset):

    def __init__(self, lambda_, pos_class_label=0, num_of_images=None, inverse=False, batch_size=32):
        super(MMCFDataset, self).__init__(
            pos_class_label=pos_class_label, num_of_images=num_of_images, inverse=inverse, batch_size=batch_size
        )
        self.lambda_ = lambda_

    def prepare_data_from_array(self, train_images, train_labels, num_of_images_last=False):

        if not 3 <= len(train_images.shape) <= 4:
            raise ValueError(
                'Train data shape should be 3 or 4 dimensional! Got {}-dimensional data array!'.format(
                                                                                                 len(train_images.shape)
                )
            )
        if num_of_images_last:
            train_images = np.moveaxis(train_images, -1, 0)

        if not self.num_of_images or self.num_of_images > train_images.shape[0]:
            self.num_of_images = train_images.shape[0]
            print('num_of_images set to {}!'.format(self.num_of_images))

        if self.gt_label not in train_labels:
            raise ValueError('Incorrect gt_label {}! Classes available: {}'.format(self.gt_label,
                                                                                   np.unique(train_labels)))

        train_images, train_labels = self._balance_class(images=train_images, labels=train_labels)

        print('num_of_images set to {}!'.format(train_images.shape[0]))
        if len(np.shape(train_images)) > 3:
            train_images = np.squeeze(np.array(train_images), axis=-1)

        train_data, S = self.tf_data_transform(x_train=train_images, lambda_=self.lambda_)
        return train_data, train_labels, train_images.shape, S

    def prepare_data_from_directory(self, train_path, train_labels_path=None, target_size=(100, 100)):
        if train_labels_path:
            image_names, labels, = self.get_data_from_csv(train_labels_path, train_path)
        else:
            image_names, labels = self.get_data_from_directory(train_path)

        if not self.num_of_images or self.num_of_images > len(image_names):
            self.num_of_images = len(image_names)

        if self.gt_label not in labels:
            raise ValueError('Incorrect gt_label {}! Classes available: {}'.format(self.gt_label, np.unique(labels)))

        image_names, labels = self._balance_class(images=image_names, labels=labels)
        images = []
        for path in enumerate(image_names):
            images.append(self._get_image(path[1], target_size=target_size))

        print('num_of_images set to {}!'.format(np.shape(images)[0]))

        if len(np.shape(images)) > 3:
            images = np.squeeze(np.array(images), axis=-1)

        train_data, S = self.tf_data_transform(x_train=images, lambda_=self.lambda_)
        return np.nan_to_num(np.array(train_data)), labels, np.shape(images), S

    def tf_data_transform(self, x_train, lambda_):
        x_train = self.standard_scaler(x_train)
        x_train = self.tf_fft2(tf.cast(x_train, dtype='complex64'))

        I = tf.ones((x_train.shape[1] * x_train.shape[2], 1), dtype='complex64')
        X = tf.transpose(tf.reshape(x_train, shape=[x_train.shape[0], x_train.shape[1] * x_train.shape[2]]))
        S = (lambda_ * I + (1 - lambda_) * tf.reduce_sum(X * tf.math.conj(X), axis=1, keepdims=True)) ** (-0.5)
        print(S.shape, X.shape)
        data = S * X
        print(data.shape)
        data_train = tf.math.real(data).numpy()
        return tf.transpose(data_train), tf.transpose(S).numpy()

    @staticmethod
    def tf_fft2(image):
        """
        Direct 2D Fourier transform for the image.
        :param image: image array.
        :return: Fourier transform of the image.
        """
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(image, axes=(1, 2))), axes=(1, 2))


class NNCFDataset(CorrelationFilterDataset):

    def __init__(self, augmentation=None, sample_weight=(1, 1, 1)):
        super(NNCFDataset, self).__init__()
        self.augmentation = augmentation
        self.sample_weights = sample_weight

    def prepare_data_from_array(self, train_images, train_labels, validation_images=None, validation_labels=None,
                                num_of_images_last=False):

        if not validation_images.any():
            train_images, train_labels, validation_images, validation_labels = sklearn.model_selection.train_test_split(
                                                                                                train_images,
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
            train_image_names, validation_image_names, train_labels, validation_labels = \
                sklearn.model_selection.train_test_split(train_image_names,
                                                         train_labels,
                                                         shuffle=True,
                                                         test_size=0.33,
                                                         random_state=42)
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

    def _make_data_generator(self, images, labels):
        gt, sample_weight = self._make_gt_correlation(shape=images.shape, labels=labels)
        if self.augmentation:
            rotation_range, horizontal_flip, vertical_flip = self.augmentation
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.,
                                                                      rotation_range=rotation_range,
                                                                      horizontal_flip=horizontal_flip,
                                                                      vertical_flip=vertical_flip)
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)

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

