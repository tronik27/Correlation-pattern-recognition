import numpy as np
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from CF_dataset import CFDataset, MMCFDataset, NNCFDataset
from CF_metrics import CFMetric
from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from Correlation.Correlation_utils import Correlator, PlotCrossCorrelation, WorkingWithJointTransformCorrelator
from CF_ResNet import NNCFModel


class CorrelationFilter(ABC):
    def __init__(self, images, labels, gt_label, metric_names=None, inverse=False, filter_plane='spatial',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32):
        self.images = images
        self.labels = labels
        self.num_of_images = num_of_images
        self.filter_plane = filter_plane
        self.peak_classifier = peak_classifier
        self.batch_size = batch_size
        self.gt_label = gt_label
        self.metric_names = metric_names
        self.inverse = inverse
        self.m_values = []
        self.cf = np.zeros(0)
        self.name = "CF"

    @abstractmethod
    def synthesis(self):
        pass

    def test_classification_quality(self, path_to_filter, corr_type='fourier_correlation', get=False,
                                    plot_correlations_in_3d=False, plot_conf_matrix=False, use_wf=False,
                                    plot_correlations_in_2d=False, path_to_model=None, target_size=(100, 100)):
        if not self.cf:
            self.cf = np.load(path_to_filter)

        if isinstance(self.images, str):
            test_gen = CFDataset(gt_label=self.gt_label,
                                 num_of_images=self.num_of_images,
                                 inverse=self.inverse,
                                 batch_size=self.batch_size).make_test_data_from_directory(
                path=self.images,
                labels_path=self.labels,
                target_size=target_size
            )
        else:
            test_gen = CFDataset(gt_label=self.gt_label,
                                 num_of_images=self.num_of_images).make_test_data_from_array(
                images=self.images,
                batch_size=self.batch_size,
                labels=self.labels
            )

        batches = 0
        for img_batch, label_batch in test_gen:
            batches += 1
            correlations = self.correlation(img_batch[:, :, :, 0], corr_type=corr_type, use_wf=use_wf)
            if batches >= self.num_of_images / self.batch_size:
                self.calculate_metric_values(labels=label_batch,
                                             preds=correlations,
                                             path_to_model=path_to_model,
                                             get=get,
                                             finish=True)
                break
            else:
                self.calculate_metric_values(labels=label_batch,
                                             preds=correlations,
                                             path_to_model=path_to_model,
                                             get=get)
        correlations = correlations[:, correlations.shape[1] // 2 - 16: correlations.shape[1] // 2 + 16,
                                    correlations.shape[2] // 2 - 16: correlations.shape[2] // 2 + 16]
        # print('c', correlations.shape)
        # print('pc', correlations[label_batch == 1].shape)
        # print('nc', correlations[label_batch != 1].shape)
        # print('gt', self.gt_label)
        # print('labels', label_batch)
        # np.save('CC/positive/yale{}_{}'.format(self.name, self.gt_label), correlations[label_batch == 1])
        # np.save('CC/negative/yale{}_{}'.format(self.name, self.gt_label), correlations[label_batch != 1])
        if plot_correlations_in_3d:
            self.plot_3d_correlations(correlations, label_batch)
        if plot_correlations_in_2d:
            self.plot_2d_correlations(correlations, label_batch)
        if plot_conf_matrix:
            self.plot_confusion_matrix(CFMetric(path_to_model=path_to_model, peak_classifier='pcnn'),
                                       label_batch, np.array(correlations))

    def correlation(self, images, corr_type, use_wf, num_images_last=False):
        c = Correlator(images, self.cf, filter_plane=self.filter_plane, num_images_last=num_images_last,
                       use_wf=use_wf)
        scenes = getattr(c, corr_type)()
        return scenes

    def make_amplitude_hologram(self, modulator_size, sample_level, path_to_save, multi=False):
        c = WorkingWithJointTransformCorrelator(path_to_save=path_to_save, path_to_filter=self.cf,
                                                path_to_data=None, modulator_size=modulator_size)
        c.cf = self.cf
        c.cf_hologram_recovery(holo_name=self.name, sample_level=sample_level, filter_plane=self.filter_plane,
                               multi=multi)

    def calculate_metric_values(self, labels, preds, path_to_model, finish=False, get=False):
        metrics = self.get_metrics(path_to_model=path_to_model)
        if not get:
            print(self.name + ':')
        for metric in metrics:
            metric.update_state(labels, preds)
            if finish and get:
                self.m_values.append(metric.result().numpy())
            elif finish and not get:
                print('{}: {}'.format(metric.name, metric.result().numpy()))

    def get_metrics(self, path_to_model):
        metrics = []
        for metric_name in self.metric_names:
            metrics.append(CFMetric(name=metric_name, peak_classifier=self.peak_classifier, path_to_model=path_to_model))
        return metrics

    def save_matrix(self, path):
        if self.cf.any():
            np.save(path, self.cf)
        else:
            print('Could not save correlation filter. Nothing to save!')

    def show(self):
        if self.cf.any():
            plt.imshow(np.abs(self.cf), cmap='gray')
            plt.title(self.name, fontsize=24)
            plt.axis('off')
            plt.show()
        else:
            print('Could not show correlation filter. Nothing to show!')

    @staticmethod
    def get_from_center(scene, shape):
        return scene[:, (scene.shape[-2] - shape[-2]) // 2:(scene.shape[-2] + shape[-2]) // 2,
                     (scene.shape[-1] - shape[-1]) // 2:(scene.shape[-1] + shape[-1]) // 2]

    @staticmethod
    def plot_3d_correlations(correlations, labels):
        if len(labels) >= 4:
            n = 2
        else:
            n = len(labels) // 2
        indexes = np.hstack((np.random.choice(np.where(labels == 0)[0], n),
                             np.random.choice(np.where(labels == 1)[0], n)))
        selected_correlations = [correlations[i] for i in indexes]
        selected_labels = [labels[i] for i in indexes]
        PlotCrossCorrelation(corr_scenes=np.array(selected_correlations), labels=np.array(selected_labels)).plot_3D()

    @staticmethod
    def plot_2d_correlations(correlations, labels):
        if len(labels) >= 32:
            n = 16
        else:
            n = len(labels) // 2
        indexes = np.hstack((np.random.choice(np.where(labels == 1)[0], n),
                             np.random.choice(np.where(labels == 0)[0], n)))
        selected_correlations = [correlations[i] for i in indexes]
        selected_labels = [labels[i] for i in indexes]
        PlotCrossCorrelation(corr_scenes=np.array(selected_correlations), labels=np.array(selected_labels)).plot()

    @staticmethod
    def plot_confusion_matrix(m, true_labels, correlations):
        pred_labels = m.peak_classifier(correlations)
        conf_matrix = tf.math.confusion_matrix(labels=true_labels[:, 0], predictions=pred_labels[:, 0], num_classes=2)
        _, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix.numpy() / (pred_labels.shape[0] / 2), annot=True, cmap=plt.cm.Blues)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title("Confusion matrix", size=18)
        plt.show()

    @staticmethod
    def tf_fft2(image):
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))

    @staticmethod
    def tf_ifft2(image):
        return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(image, axes=(-2, -1))), axes=(-2, -1))


class NNCF(CorrelationFilter):
    def __init__(self, images, labels=None, validation_images=None, validation_labels=None,  num_of_images=12000,
                 gt_label=0, metric_names=['accuracy'], peak_classifier='pcnn', inverse=False):
        super(MMCF, self).__init__()
        self.images = images
        self.labels = labels
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.num_of_images = num_of_images
        self.gt_label = gt_label
        self.metric_names = metric_names
        self.peak_classifier = peak_classifier
        self.inverse = inverse
        self.model_summary = None
        self.test_mode = False

    def synthesis(self, num_correlations=32, epochs=1000, initial_filter_matrix=None, target_size=(100, 100),
                  sample_weight=(1, 100, 200), augmentation=(0.2, True, True), show_learning_curves=False,
                  num_of_images_last=False, plot_output_correlations=False, learning_rate=0.001, min_delta=0.0001,
                  scheduler=None):

        datagen = NNCFDataset(gt_label=self.gt_label, num_of_corr=num_correlations, sample_weight=sample_weight,
                              num_of_images=self.num_of_images, augmentation=augmentation, inverse=self.inverse)
        if isinstance(self.images, str):
            train_data, validation_data, data_len = datagen.prepare_data_from_directory(
                                                                        train_path=self.images,
                                                                        validation_path=self.validation_images,
                                                                        train_labels_path=self.labels,
                                                                        validation_labels_path=self.validation_labels,
                                                                        target_size=target_size,
                                                                        initial_filter_matrix=initial_filter_matrix
                                                                            )
            shape = (data_len, target_size[0], target_size[1], 1)
        else:
            train_data, validation_data, shape = datagen.prepare_data_from_array(
                                                                            train_images=self.images,
                                                                            train_labels=self.labels,
                                                                            validation_images=self.validation_images,
                                                                            validation_labels=self.validation_labels,
                                                                            num_of_images_last=num_of_images_last,
                                                                                )
        nn = NNCFModel(input_shape=[shape[1:], shape[1:]], num_classes=2, num_filters=16,
                       regularization=0.0005, input_name='cf').build()

        self.model_summary = nn.summary()

        nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=self.get_metrics(shape=shape)
                   )

        history = nn.fit(train_data, validation_data=validation_data, steps_per_epoch=shape[0] // num_correlations,
                         callbacks=self.get_callbacks(train_data, min_delta=min_delta, scheduler=scheduler),
                         epochs=epochs)

        best_nn = NNCFModel(input_shape=[shape[1:], shape[1:]], num_classes=2, num_filters=16,
                            regularization=0.0005, input_name='cf').build()
        best_nn.load_weights('training_nncf/weights')

        self.cf = best_nn.get_correlation_filter(data=validation_data[0])

        if plot_output_correlations:
            best_nn.plot_output_correlations(data=validation_data[0])
        if show_learning_curves:
            self.plot_learning_curves(history)


class MMCF(CorrelationFilter):

    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32, lambda_=0.5, c=1000):
        super(MMCF, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                   num_of_images, batch_size)
        self.lambda_ = lambda_
        self.c = c
        self.name = "MMCF"

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:

        datagen = MMCFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                              inverse=self.inverse, lambda_=self.lambda_)
        if isinstance(self.images, str):
            X_train_MMCF, train_labels, shape, S = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            X_train_MMCF, train_labels, shape, S = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )
        svm = SVC(kernel='linear', probability=False, C=self.c, random_state=42)
        svm.fit(X_train_MMCF, train_labels)
        alphas = svm.dual_coef_[:, train_labels[svm.support_] == 1]
        support_vectors = X_train_MMCF[svm.support_][train_labels[svm.support_] == 1, :]
        print('Количество опорных векторов:', svm.n_support_.sum())
        print('Количество опорных векторов для положительного класса:', support_vectors.shape[0])
        w = S * np.matmul(alphas, support_vectors)
        self.cf = w.reshape((shape[1], shape[2]))
        return self.cf


class OTSDF(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32, lambda_=0.5):
        super(OTSDF, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                    num_of_images, batch_size)
        self.lambda_ = lambda_
        self.name = 'OTSDF'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        OTSDF correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape
        x_matrix = self.tf_fft2(tf.cast(train_images, dtype='complex64'))
        X = tf.transpose(tf.reshape(x_matrix, shape=[img_number, height * width]))
        D = tf.reduce_sum(X * tf.math.conj(X), axis=1, keepdims=True) / img_number
        T = (self.lambda_ * D + (1 - self.lambda_) * tf.ones_like(D))/(self.lambda_ + 1e-7)
        g = tf.ones((img_number, 1), dtype='complex64')
        h = tf.linalg.matmul(tf.linalg.matmul(T**(-1) * X, tf.linalg.inv(tf.linalg.matmul(tf.math.conj(X) * T**(-1), X,
                                                                                          transpose_a=True))), g)
        self.cf = tf.reshape(h, shape=[height, width]).numpy()
        return self.cf


class MVSDF(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32):
        super(MVSDF, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                    num_of_images, batch_size)
        self.name = 'MVSDF'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        MVSDF correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        self.cf = OTSDF(images=self.images,
                        labels=self.labels,
                        gt_label=self.gt_label,
                        metric_names=self.metric_names,
                        inverse=self.inverse,
                        num_of_images=self.num_of_images,
                        lambda_=0).synthesis(target_size=target_size, num_of_images_last=num_of_images_last)
        return self.cf


class MACE(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32):
        super(MACE, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                   num_of_images, batch_size)
        self.name = 'MACE'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        MACE correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        self.cf = OTSDF(images=self.images,
                        labels=self.labels,
                        gt_label=self.gt_label,
                        metric_names=self.metric_names,
                        inverse=self.inverse,
                        num_of_images=self.num_of_images,
                        lambda_=1).synthesis(target_size=target_size, num_of_images_last=num_of_images_last)
        return self.cf


class UOTSDF(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32, lambda_: float = 0.5):
        super(UOTSDF, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                     num_of_images, batch_size)
        self.name = 'UOTSDF'
        self.lambda_ = lambda_

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        UOTSDF correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape
        x_matrix = self.tf_fft2(tf.cast(train_images, dtype='complex64'))
        X = tf.transpose(tf.reshape(x_matrix, shape=[img_number, height * width]))
        D = tf.reduce_sum(X * tf.math.conj(X), axis=1, keepdims=True) / img_number
        M = tf.reduce_mean(X, axis=1, keepdims=True)
        S = tf.reduce_sum((X - M) * tf.math.conj(X - M), axis=1, keepdims=True) / img_number
        h = M / (self.lambda_ * D + (1 - self.lambda_) * S)
        self.cf = tf.reshape(h, shape=[height, width]).numpy()
        return self.cf


class MACH(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32):
        super(MACH, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                   num_of_images, batch_size)
        self.name = 'MACH'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        UOTSDF correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        self.cf = UOTSDF(images=self.images,
                         labels=self.labels,
                         gt_label=self.gt_label,
                         metric_names=self.metric_names,
                         inverse=self.inverse,
                         num_of_images=self.num_of_images,
                         lambda_=0).synthesis(target_size=target_size, num_of_images_last=num_of_images_last)
        return self.cf


class UMACE(CorrelationFilter):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32):
        super(UMACE, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                    num_of_images, batch_size)
        self.name = 'UMACE'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False) -> np.ndarray:
        """
        UMACE correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :return: obtained OT MACH correlation filter.
        """
        self.cf = UOTSDF(images=self.images,
                         labels=self.labels,
                         gt_label=self.gt_label,
                         metric_names=self.metric_names,
                         inverse=self.inverse,
                         num_of_images=self.num_of_images,
                         lambda_=1).synthesis(target_size=target_size, num_of_images_last=num_of_images_last)
        return self.cf


class ASEF(CorrelationFilter):
    def __init__(self, images, labels, gt_label, metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32, fwhm: int = 1):
        super(ASEF, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                   num_of_images, batch_size)
        self.name = 'ASEF'
        self.fwhm = fwhm

    def synthesis(self, target_size=(100, 100), num_of_images_last=False, peak_coordinates: np.ndarray = np.zeros(1))\
            -> np.ndarray:
        """
        ASEF correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :param peak_coordinates: gt coordinates of correlation peak.
        :return: obtained OT MACH correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape

        x_matrix = self.tf_fft2(tf.cast(train_images, dtype='complex64'))
        X = tf.reshape(x_matrix, shape=[img_number, height * width])
        g = self.make_gt_correlation(peak_coordinates=peak_coordinates, shape=(height, width))
        D = X * tf.math.conj(X)
        p = X * g
        h = tf.reduce_sum(p / (D + 1), axis=0, keepdims=True) / img_number
        self.cf = tf.reshape(h, shape=[height, width]).numpy()
        return self.cf

    def make_gt_correlation(self, shape, peak_coordinates):
        x = np.arange(0, shape[0], 1, float)
        y = x[:, np.newaxis]

        if not peak_coordinates.any():
            x0 = y0 = shape[0] // 2
        else:
            x0 = peak_coordinates[0]
            y0 = peak_coordinates[1]

        gt_corr = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / self.fwhm ** 2)
        gt_corr = tf.reshape(self.tf_fft2(tf.cast(gt_corr, dtype='complex64')), shape=[shape[0] * shape[1]])
        return gt_corr


class MOSSE(ASEF):
    def __init__(self, images, gt_label, labels,  metric_names=None,  inverse=False, filter_plane='freq',
                 peak_classifier='pcnn', num_of_images=1000, batch_size=32, fwhm=1):
        super(MOSSE, self).__init__(images, labels, gt_label, metric_names, inverse, filter_plane, peak_classifier,
                                    num_of_images, batch_size)
        self.fwhm = fwhm
        self.name = 'MOSSE'

    def synthesis(self, target_size=(100, 100), num_of_images_last=False, peak_coordinates: np.ndarray = np.zeros(1))\
            -> np.ndarray:
        """
        MOSSE correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :param peak_coordinates: gt coordinates of correlation peak.
        :return: obtained MOSSE correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        print('labels', self.labels)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape

        x_matrix = self.tf_fft2(tf.cast(train_images, dtype='complex64'))
        X = tf.reshape(x_matrix, shape=[img_number, height * width])
        g = self.make_gt_correlation(peak_coordinates=peak_coordinates, shape=(height, width))
        D = tf.reduce_sum(X * tf.math.conj(X), axis=0, keepdims=True) / (img_number * height * width)
        p = tf.reduce_sum(X * g, axis=0, keepdims=True) / (img_number * height * width)
        h = p / (D + 1e-7)
        self.cf = tf.reshape(h, shape=[height, width]).numpy()
        return self.cf


class MINACE(CorrelationFilter):

    def synthesis(self, target_size=(100, 100), num_of_images_last=False, noise_level: int = 2, nu: float = 0.0) \
            -> np.ndarray:
        """
            MINACE correlation filter.
            :param num_of_images_last: axis responsible for the number of images in the set.
            :param noise_level: noise level for noise matrix.
            :param nu: noise matrix coefficient.
            :param target_size: desired filter size.
            :return: obtained MINACE correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape

        x_matrix = self.fft2(train_images)
        x_matrix = np.reshape(x_matrix, (img_number, height * width))

        d_matrix = x_matrix * np.conj(x_matrix)

        noise_matrix = self.fft2(np.random.randint(1, noise_level, (height, width))) ** 2
        noise_matrix = np.reshape(noise_matrix, (1, height * width))

        t_matrix = np.max(np.concatenate([d_matrix, nu * noise_matrix], axis=0), axis=0)
        t_matrix = t_matrix ** (-1)

        # MINACE formula: H = T^(-1) * X * (X^(+)*T^(-1)*X)^(-1) * c.
        h_matrix = np.dot(x_matrix, np.conj(x_matrix.T) * np.expand_dims(t_matrix, axis=1))
        h_matrix = np.dot(np.linalg.inv(h_matrix), np.expand_dims(t_matrix, axis=0) * x_matrix)
        h_matrix = np.dot(np.ones(img_number), h_matrix)
        self.cf = np.reshape(h_matrix, (height, width))
        return self.cf

    @staticmethod
    def fft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
        """
        Direct 2D Fourier transform for the image.
        :param image: image array.
        :param axes: axes over which to compute the FFT.
        :return: Fourier transform of the image.
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)


class OT_MACH(CorrelationFilter):

    def synthesis(self, target_size=(100, 100), num_of_images_last=False, alpha: float = 1.0, beta: float = 1.0)\
            -> np.ndarray:
        """
        OT MACH correlation filter.
        :param num_of_images_last: axis responsible for the number of images in the set.
        :param target_size: desired filter size.
        :param alpha: S matrix coefficient.
        :param beta: D matrix coefficient.
        :return: obtained OT MACH correlation filter.
        """
        datagen = CFDataset(gt_label=self.gt_label, num_of_images=self.num_of_images,
                            inverse=self.inverse)
        if isinstance(self.images, str):
            train_images = datagen.prepare_data_from_directory(
                train_path=self.images,
                train_labels_path=self.labels,
                target_size=target_size
            )
        else:
            train_images = datagen.prepare_data_from_array(
                train_images=self.images,
                train_labels=self.labels,
                num_of_images_last=num_of_images_last,
            )

        img_number, height, width = train_images.shape

        x_matrix = self.tf_fft2(tf.cast(train_images, dtype='complex64'))
        x_matrix = tf.reshape(x_matrix, shape=[img_number, height * width])

        mean_vector = tf.reduce_sum(x_matrix, axis=0) / img_number
        mean_vector = tf.expand_dims(mean_vector, axis=0)

        s_matrix = tf.reduce_sum((x_matrix - mean_vector) * tf.math.conj(x_matrix - mean_vector), axis=0) / img_number

        d_matrix = tf.reduce_sum(x_matrix * tf.math.conj(x_matrix), axis=0) / img_number

        h_matrix = (alpha * s_matrix + beta * d_matrix) ** (-1)
        h_matrix *= mean_vector[0, :]
        self.cf = tf.reshape(h_matrix, shape=[height, width]).numpy()
        return self.cf
