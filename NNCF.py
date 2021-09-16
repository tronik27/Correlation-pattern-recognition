import tensorflow as tf
import matplotlib.pyplot as plt
from Correlation.Correlation_utils import Correlator, PlotCrossCorrelation
from nncf_utils import FilterDataset, make_tensorboard, CFMetric
from CF_ResNet import NNCFModel
import numpy as np
import seaborn as sns


class NNCF:
    def __init__(self, images, labels=None, validation_images=None, validation_labels=None,  num_of_images=12000,
                 gt_label=0, metric_names=['accuracy'], peak_classifier='pcnn', inverse=False):
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
        self.cf = None
        self.test_mode = False

    def synthesis(self, num_correlations=32, epochs=1000, initial_filter_matrix=None, target_size=(100, 100),
                  sample_weight=(1, 100, 200), augmentation=(0.2, True, True), show_learning_curves=False,
                  num_of_images_last=False, plot_output_correlations=False, learning_rate=0.001, min_delta=0.0001,
                  scheduler=None):

        datagen = FilterDataset(gt_label=self.gt_label, num_of_corr=num_correlations, sample_weight=sample_weight,
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

    def test_classification_quality(self, path_to_filter='nncf.npy', correlation_type='fourier_correlation',
                                    plot_correlations_in_3d=False, plot_conf_matrix=False,
                                    plot_correlations_in_2d=False, modulator_size=(1080, 1920)):

        self.test_mode = True
        if not self.cf:
            self.cf = np.load(path_to_filter)

        if isinstance(self.images, str):
            test_images, test_labels = FilterDataset(gt_label=self.gt_label,
                                                     num_of_images=self.num_of_images).make_test_data_from_directory(
                                                                                train_path=self.images,
                                                                                validation_path=self.validation_images
                                                                                                     )
        else:
            test_images, test_labels = FilterDataset(gt_label=self.gt_label,
                                                     num_of_images=self.num_of_images).make_test_data_from_array(
                                                                                                    images=self.images,
                                                                                                    labels=self.labels
                                                                                                     )

        correlations = []
        for image in test_images:
            cross_correlation_method = getattr(Correlator(image=image, cf=self.cf, modulator_size=modulator_size),
                                               correlation_type)
            correlations.append(cross_correlation_method())

        print('test_images:', test_images.shape)
        print('cf:', self.cf.shape)
        print('correlations:', np.array(correlations).shape)

        self.calculate_metric_values(test_images.shape, test_labels, np.array(correlations))
        if plot_correlations_in_3d:
            self.plot_3d_correlations(correlations, test_labels)
        if plot_correlations_in_2d:
            self.plot_2d_correlations(correlations, test_labels)
        if plot_conf_matrix:
            self.plot_confusion_matrix(CFMetric(shape=test_images.shape), test_labels, np.array(correlations))

    def test_recognition_quality(self, path_to_filter='nncf.npy'):
        pass

    def save_matrix(self, path='nncf'):
        if self.cf.any():
            np.save(path, self.cf)
        else:
            print('Could not save correlation filter. Nothing to save!')

    def show(self):
        if self.cf.any():
            plt.imshow(self.cf, cmap='gray')
            plt.show()
        else:
            print('Could not show correlation filter. Nothing to show!')

    def calculate_metric_values(self, shape, labels, preds):
        metrics = self.get_metrics(shape)
        for metric in metrics:
            metric.update_state(labels, preds)
            print('{}: {}'.format(metric.name, metric.result().numpy()))

    # def get_metrics(self, shape):
    #     metrics = []
    #     for metric_name in self.metric_names:
    #         metrics.append(CFMetric(shape=shape, name=metric_name,
    #                                 peak_classifier=self.peak_classifier, test_mode=self.test_mode))
    #     return metrics

    @staticmethod
    def get_metrics(shape):
        metrics = list()
        metrics.append(tf.keras.metrics.BinaryAccuracy(threshold=0.5))
        metrics.append(tf.keras.metrics.Precision())
        metrics.append(tf.keras.metrics.Recall())
        return metrics

    @staticmethod
    def get_callbacks(data, min_delta, scheduler=None):
        callbacks = []
        if scheduler:
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callbacks.append(lr_scheduler)

        tensorboard_callback = make_tensorboard()
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.0001,
                                                         factor=0.1, patience=3, min_lr=0.00001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('training_nncf/weights', save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=min_delta)
        # callbacks += [tensorboard_callback, reduce_lr, set_weights, checkpoint, stop]
        callbacks += [tensorboard_callback, reduce_lr, checkpoint, stop]
        return callbacks

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
    def plot_confusion_matrix(m, test_labels, correlations):
        true_labels, pred_labels = m.correlation_classification(test_labels, correlations)
        conf_matrix = tf.math.confusion_matrix(labels=true_labels[:, 0], predictions=pred_labels[:, 0], num_classes=2)
        _, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix.numpy() / (pred_labels.shape[0] / 2), annot=True, cmap=plt.cm.Blues)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title("Confusion matrix", size=18)
        plt.show()

    @staticmethod
    def plot_learning_curves(metriks_data):
        pass
