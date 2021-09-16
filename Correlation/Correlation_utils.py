import numpy as np
import cv2 as cv
from numpy.fft import ifft2
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.filters import threshold_yen
from skimage.feature import hog
import tensorflow as tf
from skimage.transform import resize
from typing import Tuple
from math import ceil
from tkinter import filedialog
from tkinter import messagebox
import rawpy
from tkinter.filedialog import asksaveasfilename
import os


class Correlator:
    def __init__(self, images, cf, filter_plane='spatial', num_images_last=False, use_wf=False):
        if len(cf.shape) == 2:
            self.cf = tf.expand_dims(tf.cast(cf, dtype='complex64'), axis=0)
        elif len(cf.shape) == 3:
            self.cf = tf.cast(cf, dtype='complex64')
        else:
            raise ValueError('Incorrect rank of correlation filter array! Should be 2 or 3, found {}!'.format(
                len(cf.shape)))

        if len(images.shape) == 2:
            self.images = tf.expand_dims(tf.cast(images, dtype='complex64'), axis=0)
        elif len(images.shape) == 3:
            self.images = tf.cast(images, dtype='complex64')
        else:
            raise ValueError('Incorrect rank of images array! Should be 2 or 3, found {}!'.format(len(images.shape)))

        if num_images_last:
            self.images = tf.transpose(self.images, perm=[0, 2, 1])
        if filter_plane == 'freq':
            self.cf = self.tf_ifft2(self.cf)

        if self.cf.shape[1] > self.images.shape[1] or self.cf.shape[2] > self.images.shape[2]:
            raise ValueError('Filter is bigger than image! Found image size: {}, should be at least:{}!'.format(
                self.images.shape[1:], self.cf.shape))
        self.use_wf = use_wf
        self.scenes = tf.zeros(1)

    def correlation(self):

        corr_filter = tf.expand_dims(self.cf, axis=3)

        image = np.transpose(np.expand_dims(self.images - np.mean(self.images), axis=(2, 3)), (3, 0, 1, 2))
        tensor = tf.convert_to_tensor(image, dtype='float32')
        cf = tf.convert_to_tensor(corr_filter - np.mean(corr_filter), dtype='float32')
        self.scenes = np.abs(tf.nn.conv2d(tensor, cf, strides=[1, 1, 1, 1], padding='SAME')[0, :, :, 0].numpy())

        return self.scenes

    def fourier_correlation(self):
        cf, img = self._prepare(use_wf=self.use_wf)
        corr = tf.abs(self.tf_ifft2(self.tf_fft2(img) * tf.math.conj(self.tf_fft2(cf)))).numpy()
        self.scenes = self._get_from_center(scene=corr, shape=self.images.shape)
        return self.scenes

    def van_der_lugt(self, sample_level=8):
        cf, img = self._put_on_modulator(correlator_type='4f', sample_level=sample_level)
        plt.imshow(np.abs(img.numpy()[0, :, :]), cmap='gray')
        plt.show()
        self.scenes = tf.math.abs(self.tf_ifft2(self.tf_fft2(img) * np.conj(cf))).numpy()
        self._get_peak(size_param=0.5, correlator_type='4f')
        return self.scenes

    def joint_transform(self, size_param=0.1, sample_level=8):
        img = self._put_on_modulator(correlator_type='2f', sample_level=sample_level)
        self.scenes = tf.math.abs(self.tf_ifft2(tf.cast(tf.abs(self.tf_fft2(img)), dtype='complex64'))).numpy()

        self._get_peak(size_param=size_param, correlator_type='2f')
        return self.scenes

    def _prepare(self, use_wf=False):
        if use_wf:
            img = self._window_function(self.images)
            cf = self.cf
        else:
            img = self.zero_mean(self.images)
            paddings = tf.constant([[0, 0], [self.cf.shape[-2] // 2, self.cf.shape[-2] // 2],
                                    [self.cf.shape[-1] // 2, self.cf.shape[-1] // 2]])
            img = tf.pad(img, paddings, "CONSTANT")

            cf = self.zero_mean(self.cf)
            paddings = tf.constant([[0, 0], [self.images.shape[-2] // 2, self.images.shape[-2] // 2],
                                    [self.images.shape[-1] // 2, self.images.shape[-1] // 2]])
            cf = tf.pad(cf, paddings, "CONSTANT")
        return cf, img

    def _put_on_modulator(self, correlator_type, sample_level):
        cf, images = self._prepare_data_for_modulator(correlator_type=correlator_type, sample_level=sample_level)
        if correlator_type == '2f':
            paddings = tf.constant([[0, 0], [10, images.shape[1] * 3 - 10],
                                    [10, images.shape[2] * 3 - 10]])
            img1 = tf.pad(images, paddings, "CONSTANT")
            paddings = tf.constant([[0, 0], [images.shape[1] * 4 - 10 - cf.shape[1], 10],
                                    [images.shape[2] * 4 - 10 - cf.shape[2], 10]])
            img2 = tf.pad(cf, paddings, "CONSTANT")
            img = img1 + img2
            return img
        else:
            mean_energy = tf.cast((tf.reduce_mean(images) + tf.reduce_mean(cf)) / 2, dtype='int8').numpy()
            paddings = tf.constant([[0, 0], [self.images.shape[1] // 2, self.images.shape[1] * 5 // 2],
                                    [self.images.shape[2] // 2, self.images.shape[2] * 5 // 2]])
            img = tf.pad(images, paddings, "CONSTANT", constant_values=0)
            paddings = tf.constant([[0, 0], [(img.shape[1] - cf.shape[1]) // 2, (img.shape[1] - cf.shape[1]) // 2],
                                    [(img.shape[2] - cf.shape[2]) // 2, (img.shape[2] - cf.shape[2]) // 2]])
            cf = tf.pad(cf, paddings, "CONSTANT", constant_values=0)
            return cf, img

    def _get_peak(self, size_param, correlator_type):
        y, x = self.scenes.shape[1] // 2, self.scenes.shape[2] // 2
        y1, x1 = self.images.shape[1:]
        print(self.scenes.shape)
        if correlator_type == '2f':
            self.scenes[:, :, x - int(size_param * x1): x + int(size_param * x1)] = 0
            self.scenes[:, y - int(size_param * y1): y + int(size_param * y1), :] = 0
        else:
            self.scenes[:, y - int(size_param * y1): y + int(size_param * y1),
                        x - int(size_param * x1): x + int(size_param * x1)] = 0
        # self.scenes = self.scenes[:, 400: 800, 400: 800]
        # first_maximum_coordinates = np.unravel_index(np.argmax(self.scenes[0, :, :]), self.scenes.shape[1:])
        # y, x = first_maximum_coordinates[0], first_maximum_coordinates[1]
        # print('1', y, x)
        # self.scenes = self.scenes[:, y - y1 // 2: y + y1 // 2, x - x1 // 2: x + x1 // 2]

    def _prepare_data_for_modulator(self, correlator_type, sample_level=8):
        if correlator_type == '4f':
            cf = self._amplitude_hologram(sample_level=sample_level)
            img = self.quantization(sample_level=sample_level, x=self.images)
        else:
            img = self.quantization(sample_level=sample_level, x=self.images)
            cf = self.quantization(sample_level=sample_level, x=self.cf)
        return cf, img

    def _amplitude_hologram(self, sample_level=8):
        paddings = tf.constant([[0, 0], [0, self.cf.shape[1]],
                                [0, self.cf.shape[2]]])
        H = tf.pad(self.zero_mean(self.cf), paddings, "CONSTANT")
        H = self.tf_fft2(H)
        holo = tf.math.real(H) - tf.reduce_min(tf.math.real(H), axis=(-2, -1), keepdims=True)
        holo = self.quantization(sample_level=sample_level, x=holo)
        return holo

    def plot(self):
        if self.scenes.any():
            plt.figure(figsize=(8, 3))
            ax1 = plt.subplot(1, 3, 1, adjustable='box')
            ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box')
            ax3 = plt.subplot(1, 3, 3)

            ax1.imshow(self.images[0, :, :], cmap='gray')
            ax1.set_axis_off()
            ax1.set_title('Scene with object')

            ax2.imshow(np.abs(self.cf[0, :, :]), cmap='gray')
            ax2.set_axis_off()
            ax2.set_title('Correlation Filter')

            ax3.imshow(np.abs(self.scenes[0, :, :]))
            ax3.set_axis_off()
            ax3.set_title("Cross-correlation")
            plt.show()
        else:
            print('Nothing to show!')

    @staticmethod
    def zero_mean(x):
        return x - tf.reduce_mean(x, axis=(-2, -1), keepdims=True)

    @staticmethod
    def quantization(sample_level, x):
        x = tf.abs(x)
        quant = tf.cast(x / tf.reduce_max(x, axis=(-2, -1), keepdims=True), dtype='int8') * (2 ** sample_level - 1)
        return tf.cast(quant, dtype='complex64')

    @staticmethod
    def _get_from_center(scene, shape):
        if shape[0] > scene.shape[0] or shape[1] > scene.shape[1]:
            raise ValueError("The image should be smaller than the scene on which it is placed!")
        return scene[:, (scene.shape[1] - shape[1]) // 2:(scene.shape[1] + shape[1]) // 2,
                     (scene.shape[2] - shape[2]) // 2:(scene.shape[2] + shape[2]) // 2]

    @staticmethod
    def tf_fft2(image):
        """
        Direct 2D Fourier transform for the image.
        :param image: image array.
        :return: Fourier transform of the image.
        """
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(image)))

    @staticmethod
    def tf_ifft2(image):
        """
        Inverse 2D Fourier Transform for the image.
        :param image: image array.
        :return: Fourier transform of the image.
        """
        return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(image)))

    @staticmethod
    def _window_function(image):
        x = np.arange(0, image.shape[-1], 1, float)
        y = x[:, np.newaxis]
        window = np.sin(np.pi * x / image.shape[-2]) * np.sin(np.pi * y / image.shape[-1])
        return image * window


class WorkingWithJointTransformCorrelator:

    def __init__(self, path_to_save=None, path_to_filter=None, path_to_data=None, modulator_size=(1080, 1920)):
        self.path_to_data = path_to_data
        self.path_to_filter = path_to_filter
        self.cf = np.zeros((1, 1))
        self.path_to_save = path_to_save
        self.modulator_size = modulator_size

    def prepare_data(self, inverse_image=False, filter_plane='spatial'):

        self._get_cf(filter_plane)

        if not self.path_to_data:
            messagebox.showinfo("SRS CORRELATIONS", 'Выберите изображения.')
            self.path_to_data = filedialog.askopenfilenames()

        if isinstance(self.path_to_data, str):
            self.path_to_data = list(map(lambda x: self.path_to_data + '/' + x, os.listdir(self.path_to_data)))

        for path in self.path_to_data:
            image_name = self.file_name(path)
            image = self._get_image(path, inverse=inverse_image)
            scene = self._put_on_modulator(sample_level=8, image=image)
            cv.imwrite(self.path_to_save + '/{}.bmp'.format(image_name), scene)

    def peaks_from_joint_spectrum(self, image_shape=(100, 100), size_param=0.5,
                                  build_3d_plot=False, labels=np.zeros(3)):
        if not self.path_to_data:
            messagebox.showinfo("SRS CORRELATIONS", 'Выберите изображения.')
            self.path_to_data = filedialog.askopenfilenames()

        if isinstance(self.path_to_data, str):
            self.path_to_data = list(map(lambda x: self.path_to_data + '/' + x, os.listdir(self.path_to_data)))

        if build_3d_plot:
            scenes = []
        for path in self.path_to_data:
            image_name = self.file_name(path)
            image = self._get_raw_image(path)
            scene = self.freq_to_spatial(x=image)
            scene = self.get_peak(scene=scene, img_shape=image_shape, size_param=size_param)
            if build_3d_plot:
                scenes.append(scene)
            np.save(self.path_to_save + '/{}'.format(image_name), scene)
            cv.imwrite(self.path_to_save + '/{}.bmp'.format(image_name), self.quantization(sample_level=8, x=scene))

        if build_3d_plot:
            PlotCrossCorrelation(corr_scenes=np.array(scenes), labels=labels).plot()
            # PlotCrossCorrelation(corr_scenes=np.array(scenes), labels=labels).plot_3D()

    def joint_spectrum_processing(self, sample_level=8):
        if not self.path_to_data:
            messagebox.showinfo("SRS CORRELATIONS", 'Выберите изображения.')
            self.path_to_data = filedialog.askopenfilenames()

        if isinstance(self.path_to_data, str):
            self.path_to_data = list(map(lambda x: self.path_to_data + '/' + x, os.listdir(self.path_to_data)))

        for path in self.path_to_data:
            image_name = self.file_name(path)
            image = self._get_raw_image(path)
            scene = self._put_scene_on_modulator(sample_level=sample_level, image=image)
            cv.imwrite(self.path_to_save + '/{}.bmp'.format(image_name), scene)

    def cf_hologram_recovery(self, holo_name, sample_level, filter_plane, multi):
        if not self.cf.any() and self.path_to_filter:
            self.cf = np.load(self.path_to_filter)
        elif not self.cf.any():
            messagebox.showinfo("SRS CORRELATIONS", 'Выберите корреляционный фильтр.')
            self.path_to_filter = filedialog.askopenfilename()
            self.cf = np.load(self.path_to_filter)

        holo = self._amplitude_hologram(sample_level=sample_level, filter_plane=filter_plane, multi=multi)
        if not holo_name:
            holo_name = self.file_name(path=self.path_to_filter)
        cv.imwrite(self.path_to_save + '/{}.bmp'.format(holo_name), holo)

    def _put_on_modulator(self, sample_level, image):
        cf, image = self._prepare_data_for_modulator(sample_level=sample_level, image=image)
        # paddings = ((self.modulator_size[0] // 4 - image.shape[0] // 2,
        #              3 * self.modulator_size[0] // 4 - image.shape[0] // 2),
        #             (self.modulator_size[1] // 4 - image.shape[1] // 2,
        #              3 * self.modulator_size[1] // 4 - image.shape[1] // 2))
        paddings = ((0, self.modulator_size[0] - image.shape[0]),
                    (250, self.modulator_size[1] - image.shape[1] - 250))
        img1 = np.pad(image, paddings, 'constant', constant_values=((0, 0), (0, 0)))

        # paddings = ((3 * self.modulator_size[0] // 4 - cf.shape[0] // 2,
        #              self.modulator_size[0] // 4 - cf.shape[0] // 2),
        #             (3 * self.modulator_size[1] // 4 - cf.shape[1] // 2,
        #              self.modulator_size[1] // 4 - cf.shape[1] // 2))
        paddings = ((self.modulator_size[0] - image.shape[0], 0),
                    (self.modulator_size[1] - image.shape[1] - 250, 250))
        img2 = np.pad(cf, paddings, 'constant', constant_values=((0, 0), (0, 0)))

        return img1 + img2

    def _prepare_data_for_modulator(self, sample_level, image):
        img = self.quantization(sample_level=sample_level, x=image)
        cf = self.quantization(sample_level=sample_level, x=self.cf)
        return cf, img

    def _amplitude_hologram(self, filter_plane, sample_level, multi):
        if multi:
            d_max = min(self.modulator_size) // 2
        else:
            d_max = min(self.modulator_size)

        H = np.zeros((d_max, d_max)).astype(complex)

        if filter_plane == 'freq':
            self.cf = self.freq_to_spatial(x=self.cf)
        if self.cf.shape[0] > d_max // 2 or self.cf.shape[1] > d_max // 2:
            raise ValueError('Correlation filter size is very large! Maximum size allowed fot your LM:{}'.format(
                d_max // 2))

        H[:d_max // 2, :d_max // 2] = self.put_on_center(scene=H[:d_max // 2, :d_max // 2],
                                                         image=self.zero_mean(self.cf))
        H = self._fft2(H)
        holo = np.real(H) - np.min(np.real(H))
        holo = self.quantization(sample_level=sample_level, x=holo)
        if multi:
            holo = np.vstack((holo, holo))
            holo = np.hstack((holo, holo))
            print(holo.shape)
        return holo

    def _get_image(self, path, inverse):
        image = tf.keras.preprocessing.image.load_img(path, color_mode="grayscale", target_size=self.cf.shape,
                                                      interpolation='nearest')
        image = np.squeeze(tf.keras.preprocessing.image.img_to_array(image))
        if image.shape[0] > self.modulator_size[0] // 2 or image.shape[1] > self.modulator_size[1] // 2:
            image = self.resize_image(image, max_height=self.modulator_size[0] // 2,
                                      max_wide=self.modulator_size[1] // 2)
        if inverse:
            image = 255 - image
        image = np.log(image + 1.)
        image = (image - np.mean(image)) / np.std(image)
        return image / np.max(image)

    def _get_cf(self, filter_plane):
        if self.path_to_filter:
            self.cf = np.load(self.path_to_filter)
        else:
            messagebox.showinfo("SRS CORRELATIONS", 'Выберите корреляционный фильтр.')
            self.cf = np.load(filedialog.askopenfilename())
        if filter_plane == 'freq':
            self.cf = np.abs(self.ifft2(self.cf))
        if self.cf.shape[0] > self.modulator_size[0] // 2 or self.cf.shape[1] > self.modulator_size[1] // 2:
            self.cf = self.resize_image(self.cf, max_height=self.modulator_size[0] // 2,
                                        max_wide=self.modulator_size[1] // 2)

    def _put_scene_on_modulator(self, image, sample_level):
        if (image.shape[0] > self.modulator_size[0]) or (image.shape[1] > self.modulator_size[1]):
            print(image.shape)
            image = self._resize_image(image, self.modulator_size[0], self.modulator_size[1])
            print(image.shape)
        image = self.quantization(sample_level=sample_level, x=image)
        scene = np.zeros(self.modulator_size)
        scene = self.put_on_center(scene=scene, image=image)
        return scene

    def freq_to_spatial(self, x):
        return np.abs(self.ifft2(x))

    @staticmethod
    def zero_mean(x):
        return x - np.mean(x)

    @staticmethod
    def put_on_center(scene, image):
        if image.shape[0] > scene.shape[0] or image.shape[1] > scene.shape[1]:
            raise ValueError("The image should be smaller than the scene on which it is placed!")
        scene[(scene.shape[0] - image.shape[0]) // 2:(scene.shape[0] + image.shape[0]) // 2,
              (scene.shape[1] - image.shape[1]) // 2:(scene.shape[1] + image.shape[1]) // 2] = image
        return scene

    @staticmethod
    def _resize_image(image, max_height, max_wide):
        h, w = image.shape
        iar = w / h
        mar = max_wide / max_height
        if iar < mar:
            ar_coeff = iar / mar
        else:
            ar_coeff = 1
        if (h > max_height) and (w > max_wide):
            if w > h:
                resized_image = cv.resize(image, (max_wide, int(ar_coeff * h * float(max_wide) / w)),
                                          interpolation=cv.INTER_AREA)
            else:
                resized_image = cv.resize(image, (int(ar_coeff * w * float(max_height) / h), max_height),
                                          interpolation=cv.INTER_AREA)
        elif h > max_height:
            resized_image = cv.resize(image, (int(ar_coeff * w * float(max_height) / h), max_height),
                                      interpolation=cv.INTER_AREA)
        else:
            resized_image = cv.resize(image, (max_wide, int(ar_coeff * h * float(max_wide) / w)),
                                      interpolation=cv.INTER_AREA)
        h, w = resized_image.shape

        if h > max_height:
            resized_image = resized_image[:-1, :]
        elif w > max_wide:
            resized_image = resized_image[:, :-1]
        return resized_image

    def _resize_if_necessary(self, height, width):
        if (self.image.shape[0] > height) or (self.image.shape[1] > width):
            self.image = self.resize_image(self.image, height, width)

        if (self.cf.shape[0] > height) or (self.cf.shape[1] > width):
            self.cf = self.resize_image(self.cf, height, width)

    @staticmethod
    def quantization(sample_level, x):
        return (x / np.max(x) * (2 ** sample_level - 1)).astype(np.int)

    @staticmethod
    def get_peak(scene, img_shape, size_param=0.1):
        y, x = scene.shape[0] // 2, scene.shape[1] // 2
        y1, x1 = img_shape
        scene[:, x - int(3.2 * size_param * x1): x + int(3.2 * size_param * x1)] = 0
        scene[y - int(1 * size_param * y1): y + int(1 * size_param * y1), :] = 0
        scene = scene[10: -10, 10:-10]
        # first_maximum_coordinates = np.unravel_index(np.argmax(scene), scene.shape)
        # y, x = first_maximum_coordinates[0], first_maximum_coordinates[1]
        # scene = scene[y - y1 // 2: y + y1 // 2, x - x1 // 2: x + x1 // 2]
        return scene

    @staticmethod
    def _get_raw_image(path):
        raw = rawpy.imread(path)
        img = raw.postprocess()
        if len(img.shape) > 2:
            img = np.mean(img, axis=-1)
        return img

    @staticmethod
    def file_name(path):
        path_list = path.split('/')
        file_1 = path_list.pop()
        file_1 = file_1.split('.')
        return file_1.pop(0)

    @staticmethod
    def _fft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
        """
        Direct 2D Fourier transform for the image.
        :param image: image array.
        :param axes: axes over which to compute the FFT.
        :return: Fourier transform of the image.
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)

    @staticmethod
    def ifft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
        """
        Inverse 2D Fourier Transform for Image.
        :param image: image array.
        :param axes: axes over which to compute the FFT.
        :return: Fourier transform of the image.
        """
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)

    @staticmethod
    def resize_image(image, max_height, max_wide):
        h, w = image.shape
        if (h > max_height) and (w > max_wide):
            if w > h:
                resized_image = cv.resize(image, (max_wide, int(h * float(max_wide) / w)),
                                          interpolation=cv.INTER_AREA)
            else:
                resized_image = cv.resize(image, (int(w * float(max_height) / h), max_height),
                                          interpolation=cv.INTER_AREA)
        elif h > max_height:
            resized_image = cv.resize(image, (int(w * float(max_height) / h), max_height),
                                      interpolation=cv.INTER_AREA)
        else:
            resized_image = cv.resize(image, (max_wide, int(h * float(max_wide) / w)), interpolation=cv.INTER_AREA)
        return resized_image


class PlotCrossCorrelation:

    def __init__(self, corr_scenes, labels=np.zeros(3)):
        if len(corr_scenes.shape) == 2:
            self.corr_scenes = np.expand_dims(corr_scenes, axis=0)
        elif len(corr_scenes.shape) == 4:
            self.corr_scenes = np.mean(corr_scenes, axis=-1)
        elif len(corr_scenes.shape) == 3:
            self.corr_scenes = corr_scenes
        else:
            self.corr_scenes = corr_scenes
        self.labels = labels

    def plot_3D(self):
        fig = plt.figure(figsize=(self.corr_scenes.shape[0]*5, 4))
        fig.suptitle('Cross-correlation', fontsize=16)
        for i in range(self.corr_scenes.shape[0]):
            axes = fig.add_subplot(1, self.corr_scenes.shape[0], i+1, projection='3d')
            x = np.arange(0, self.corr_scenes[i].shape[0], 1)
            y = np.arange(0, self.corr_scenes[i].shape[1], 1)
            x, y = np.meshgrid(x, y)
            surf = axes.plot_surface(x, y, self.corr_scenes[i], rstride=ceil(self.corr_scenes[i].shape[0] / 100),
                                     cstride=ceil(self.corr_scenes[i].shape[1] / 100), cmap=cm.jet)
            fig.colorbar(surf, shrink=0.5, aspect=10)
            if self.labels.any():
                if self.labels[i] == 1:
                    axes.set_title('Positive Correlation', size=15)
                else:
                    axes.set_title('Negative Correlation', size=15)
        plt.show()

    def plot(self):
        if self.corr_scenes.shape[0] > 4:
            if self.corr_scenes.shape[0] % 4:
                self.corr_scenes = self.corr_scenes[:-3, :, :]
            number_of_cols = self.corr_scenes.shape[0] // 4
            number_of_rows = 4
        elif self.corr_scenes.shape[0] <= 4:
            number_of_rows = 1
            number_of_cols = self.corr_scenes.shape[0]

        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_cols, figsize=(4 * number_of_rows,
                                                                                      4 * number_of_cols))
        axes = np.array(axes)
        for i, axe in enumerate(axes.flat):
            axe.imshow(self.corr_scenes[i, :, :], cmap=cm.jet)
            axe.axis('off')
            if self.labels.any():
                if self.labels[i] == 1:
                    axe.set_title('Positive Correlation', size=10)
                else:
                    axe.set_title('Negative Correlation', size=10)
        plt.show()


def HogBinarization(image):
    _, image = hog(image, orientations=9, pixels_per_cell=(3, 3),
                   cells_per_block=(1, 1), visualize=True, multichannel=False)
    thresh = threshold_yen(image)
    binary = image > thresh
    bin_img = 255 * binary.astype(int) + 255
    return bin_img


def CorrScenePrepare(scene):
    quant_scene = (scene - np.quantile(scene, q=0.57))
    quant_scene[quant_scene < 0.5 * np.max(quant_scene)] = 0
    scene = np.hstack((scene, quant_scene))
    return scene
