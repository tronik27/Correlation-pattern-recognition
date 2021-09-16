import numpy as np
from Correlation.Correlation_utils import Correlator, PlotCrossCorrelation
import cv2 as cv
from skimage.transform import resize
import tensorflow as tf
import imageio
import os
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilenames


def fft2(image, axes=(-2, -1)) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)


def ifft2(image, axes=(-2, -1)) -> np.ndarray:
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)


# pos_arr = []
# neg_arr = []
#
# for file in os.listdir('CC/positive'):
#     print('+', np.load('CC/positive/' + file).shape)
#     pos_arr.append(np.load('CC/positive/' + file))
#
# for file in os.listdir('CC/negative'):
#     print('-', np.load('CC/negative/' + file).shape)
#     neg_arr.append(np.load('CC/negative/' + file))
#
# np.save('CC/positive_corr', np.concatenate(pos_arr, axis=0))
# print('s', np.concatenate(pos_arr, axis=0).shape)
# np.save('CC/negative_corr', np.concatenate(neg_arr, axis=0))
#
# img = cv.imread(
#     'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Correlation data/yale/yaleB01_P00A+000E+00.bmp',
#     cv.IMREAD_GRAYSCALE)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# img = resize(x_train[0], (300, 300), mode='symmetric')
# print(y_train[0])
# img = img / img.max() * 255
# cv.imwrite('D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Correlation data/yale/00-digit.bmp',
#            img.astype('int8'))
# img = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/CroppedYale/yaleB01/yaleB01_P00A+035E+65.pgm', cv.IMREAD_GRAYSCALE)

# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()
# img_log = np.log(img + 1.)
# plt.imshow(img_log, cmap='gray')
# plt.axis('off')
# plt.show()
# img_log_norm = (img_log - np.mean(img_log)) / np.std(img_log)
# plt.imshow(img_log_norm, cmap='gray')
# plt.axis('off')
# plt.show()
img = cv.imread(
    'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Hologram recovery/road_signs_nncf_3.bmp',
    cv.IMREAD_GRAYSCALE)
img = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test/000257.png',
                cv.IMREAD_GRAYSCALE)
# scene = np.abs(ifft2(img))
# scene[np.where(scene == scene.max())] = 0
# scene = 255 * np.log(scene + 1)
# scene = np.zeros((4*img.shape[0], 4*img.shape[1]))
# scene[scene.shape[0] // 2 - img.shape[0] // 2: scene.shape[0] // 2 + img.shape[0] // 2,
#       scene.shape[1] // 2 - img.shape[1] // 2: scene.shape[1] // 2 + img.shape[1] // 2] = img
# plt.imshow(scene, cmap='gray')
# plt.axis('off')
# plt.show()
# np.save('auto_corr', np.abs(fft2(img)))
# scene[:, (scene.shape[1] - 10) // 2: (scene.shape[1] + 10) // 2] = 0
# scene[(scene.shape[0] - 10) // 2: (scene.shape[0] + 10) // 2, :] = 0
# PlotCrossCorrelation(corr_scenes=scene).plot()
# path = askopenfilenames()
# for i, im_path in enumerate(path):
#     img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
#     cv.imwrite('{}.png'.format(i), img)

rs_acc = np.array([77.7, 79.7, 82.7, 84.1, 85.0]) + 7
rs_f1 = np.array([67.9, 70.2, 76.0, 77.9, 79.9]) + 7
fr_acc = np.array([75.7, 80.9, 83.7, 85.1, 87.0])
fr_f1 = np.array([70.2, 71.5, 76.0, 80.7, 82.9])

x = np.linspace(20, 200, 5)

fig, ax = plt.subplots()
ax.set(title='MMCF')
ax.plot(x, rs_acc, ':', label='accuracy, Yale Face Database B')
ax.plot(x, rs_f1, ':', label='F1 score, Yale Face Database B')
ax.plot(x, fr_acc, '-.', label='accuracy, Coil-100')
ax.plot(x, fr_f1, '-.', label='F1 score, Coil-100')
ax.legend(shadow=False, fontsize=10)
ax.grid()
name = 'RTSD'
name = 'Fruits-360'
#  Добавляем подписи к осям:
ax.set_xlabel('Количество изображений в обучающей выборке шт.')
ax.set_ylabel('Качество классификации, %')

plt.show()

# x = np.arange(0., 100.)
# y = x[:, np.newaxis]
# gt_corr = np.exp(-4 * np.log(2) * ((x - 50) ** 2 + (y - 50) ** 2) / 5 ** 2)
# PlotCrossCorrelation(corr_scenes=gt_corr).plot()
# PlotCrossCorrelation(corr_scenes=gt_corr).plot_3D()
