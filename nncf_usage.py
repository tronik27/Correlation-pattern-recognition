from NNCF import NNCF
from keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np


def train_on_road_signs_dataset():

    def scheduler(epoch, lr):
        if epoch <= 5:
            return lr
        else:
            return lr * tf.math.exp(-0.005 * epoch)

    initial_filter_matrix = np.random.uniform(0, 1, (48, 48))

    nncf = NNCF(
        images='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train',
        labels='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv',
        validation_images='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test',
        validation_labels='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv',
        num_of_images=12000,
        gt_label=42,
        metric_names=['accuracy', 'precision', 'recall']
               )
    nncf.synthesis(num_correlations=32, epochs=10, initial_filter_matrix=initial_filter_matrix,
                   plot_output_correlations=True, target_size=(48, 48), sample_weight=(1, 100, 20000),
                   learning_rate=0.001, min_delta=0.00001, scheduler=scheduler)
    nncf.show()
    nncf.save_matrix(path='Filter/road_signs_nncf_5')


def train_on_rock_paper_scissors_dataset():

    def scheduler(epoch, lr):
        if epoch < 1:
            return lr
        elif 2 <= epoch <= 11:
            return lr * 2
        elif 12 <= epoch <= 20:
            return lr * tf.math.exp(-0.005 * epoch)
        elif epoch == 21:
            return lr * 1.5
        else:
            return lr * tf.math.exp(-0.005 * epoch)

    nncf = NNCF(
        images='D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/train',
        validation_images='D:/MIFI/SCIENTIFIC WORK/DATASETS/Rock-Paper-Scissors/test',
        num_of_images=3000,
        gt_label=1,
        metric_names=['accuracy', 'precision', 'recall'],
        inverse=True,
               )
    initial_filter_matrix = np.random.uniform(0, 1, (128, 128))
    nncf.synthesis(num_correlations=32, epochs=1000, initial_filter_matrix=initial_filter_matrix,
                   plot_output_correlations=True, target_size=(128, 128), sample_weight=(1, 1000, 1000),
                   learning_rate=0.001, min_delta=0.0001, scheduler=None)
    nncf.show()
    nncf.save_matrix(path='rps_nncf1')


def train_on_fruits_dataset():

    def scheduler(epoch, lr):
        if epoch < 1:
            return lr
        elif 2 <= epoch <= 11:
            return lr * 2
        elif 12 <= epoch <= 20:
            return lr * tf.math.exp(-0.005 * epoch)
        elif epoch == 21:
            return lr * 1.5
        else:
            return lr * tf.math.exp(-0.005 * epoch)

    nncf = NNCF(
        images='D:/MIFI/SCIENTIFIC WORK/DATASETS/fruits-360/Train',
        num_of_images=1000,
        gt_label=0,
        metric_names=['accuracy', 'precision', 'recall'],
        inverse=True,
               )
    initial_filter_matrix = np.random.uniform(0, 0.01, (100, 100))
    # initial_filter_matrix = np.zeros((100, 100))
    nncf.synthesis(num_correlations=32, epochs=1000, initial_filter_matrix=initial_filter_matrix,
                   plot_output_correlations=True, target_size=(100, 100), sample_weight=(1, 100, 1000),
                   learning_rate=0.0001, min_delta=0.0001, scheduler=None)
    nncf.show()
    nncf.save_matrix(path='Filter/fruit_nncf')


def train_on_fashion_mnist_dataset():
    (train_images, train_labels), (val_images, val_labels) = fashion_mnist.load_data()
    nncf = NNCF(images=train_images,
                labels=train_labels,
                validation_images=val_images,
                validation_labels=val_labels,
                num_of_images=12000,
                gt_label=42,
                metric_names=['accuracy', 'precision', 'recall'])
    nncf.synthesis(num_correlations=32, epochs=1000, initial_filter_matrix=None, plot_output_correlations=True,
                   sample_weight=(1, 100, 200), augmentation=(0.2, True, True))
    nncf.show()
    nncf.save_matrix()


def test_on_fashion_mnist_dataset():
    (_, _), (val_images, val_labels) = fashion_mnist.load_data()

    nncf = NNCF(images=val_images, labels=val_labels, num_of_images=2000, gt_label=1,
                metric_names=['accuracy', 'precision', 'recall'], peak_classifier='peak_position')
    nncf.test_classification_quality(correlation_type='fourier_correlation',
                                     plot_correlations_in_3d=True,
                                     plot_correlations_in_2d=True)


train_on_road_signs_dataset()
