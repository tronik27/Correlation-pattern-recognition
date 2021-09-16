from Correlation.Correlation_utils import Correlator, PlotCrossCorrelation, WorkingWithJointTransformCorrelator
import skimage
from skimage.transform import resize
from nncf_utils import FilterDataset
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from tkinter import filedialog


def fft2(image: np.ndarray, axes=(-2, -1)):
    """
    Direct 2D Fourier transform for the image.
    :param image: image array.
    :param axes: axes over which to compute the FFT.
    :return: Fourier transform of the image.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)


def ifft2(image, axes=(-2, -1)):
    """
    Inverse 2D Fourier Transform for Image.
    :param image: image array.
    :param axes: axes over which to compute the FFT.
    :return: Fourier transform of the image.
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)


def corr_test():
    img = skimage.data.camera()
    img = resize(img, (100, 100), anti_aliasing=True)
    c = Correlator(image=img, cf=img)
    scene = c.joint_transform()
    c.plot()
    PlotCrossCorrelation(corr_scenes=scene).plot()


def raw_img():
    path = 'D:/MIFI/SCIENTIFIC WORK/PHOTO/106___03/IMG_1311.CR2'
    raw = rawpy.imread(path)
    img = raw.postprocess()
    img = np.mean(img, axis=-1)
    plt.imshow(img, cmap='gray')
    plt.show()


def corr_filter_test():

    image1 = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test/000257.png',
                       cv.IMREAD_GRAYSCALE)
    image2 = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test/000181.png',
                       cv.IMREAD_GRAYSCALE)
    image = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/Tanks/Train/Chieftain-256-55-60/0001.PNG', cv.IMREAD_GRAYSCALE)
    # image = cv.imread('D:/MIFI/SCIENTIFIC WORK/DATASETS/fruits-360/Train/Banana/0400.jpg', cv.IMREAD_GRAYSCALE)
    cf = np.load('Filter/road_signs_nncf_3.npy')
    # image = resize(image1, (128, 128), anti_aliasing=True)
    # cf = resize(cf, (128, 128), anti_aliasing=True)
    # image1 = 255 - image1
    # image2 = 255 - image2
    c = Correlator(image=image, cf=image, modulator_size=(1080, 1920))
    scene = c.fourier_correlation()
    # scene = c.van_der_lugt()
    # scene = c.correlation()
    # scene = c.joint_transform(size_param=0.7)
    # c.plot()
    # print(np.where(scene == scene.max()))
    PlotCrossCorrelation(corr_scenes=scene).plot()
    PlotCrossCorrelation(corr_scenes=scene).plot_3D()
    # np.save('gt_correlation', scene / scene.max())


def dataset_test1():
    FilterDataset(gt_label=1,
                  num_of_images=64).prepare_data_from_directory(
        train_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train',
        train_labels_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
    )


def dataset_test2():
    traingen, validation, _ = FilterDataset(gt_label=1, num_of_images=64).prepare_data_from_directory(
        train_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train'
    )
    X, y, sw = traingen[0]
    print(X[0].shape, X[1].shape, y.shape, sw.shape)


def joint_transform_working():
    path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Correlation data'
    path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Hologram recovery'
    # path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FOR CORRELATOR/Joint spectrum'
    path_to_save = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FROM CORRELATOR/Correlation peaks'
    filter_path = 'Filter/road_signs_nncf_3.npy'
    path_to_data = 'D:/MIFI/SCIENTIFIC WORK/JOINT TRANSFORM/DATA FROM CORRELATOR/Joint spectrum'
    c = WorkingWithJointTransformCorrelator(path_to_save=path_to_save, path_to_filter=filter_path,
                                            path_to_data=None, modulator_size=(1080, 1920))
    # c.prepare_data(inverse_image=False, filter_plane='spatial', target_size=(100, 100))
    c.peaks_from_joint_spectrum(image_shape=(200, 200), build_3d_plot=True, labels=np.zeros(3), size_param=1)
    # c.joint_spectrum_processing(sample_level=8)
    # c.cf_hologram_recovery(sample_level=8, filter_plane='spatial')


def s_nir():
    x = np.zeros((50, 50))
    x1 = np.zeros((50, 50))
    x1[x1.shape[0]//2 - 2:x1.shape[0]//2 + 2, x1.shape[1]//2 - 2:x1.shape[1]//2 + 2] = 1
    scene = np.array([x, x1])
    scene = np.random.uniform(0, 1, (100, 100))
    PlotCrossCorrelation(corr_scenes=scene).plot()
    # PlotCrossCorrelation(corr_scenes=scene).plot_3D()


# s_nir()
# joint_transform_working()
corr_filter_test()
# path_to_save = 'D:/MIFI/SCIENTIFIC WORK/DATA FOR CORRELATOR/JOINT TRANSFORM/RPS'
# WorkingWithJointTransformCorrelator(path_to_save, inverse=True).prepare_data()
