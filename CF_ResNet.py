from abc import ABC
from typing import Tuple, Optional
from tensorflow.keras.layers import Reshape, Layer, BatchNormalization, Conv2D, Dense, LeakyReLU, Add, \
    GlobalAveragePooling2D, Activation, Input, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from Correlation.Correlation_utils import PlotCrossCorrelation
import tensorflow_addons as tfa
import numpy as np
import keras2onnx


class ResidualBlock(Layer):

    def __init__(self, num_filters, regularization, stride=1, **kwargs):
        """
        Residual block. If stride == 1, then there are no any transformations in one of the branches.
        If stride > 1, then there are convolution with 1x1 filters in one of the branches.
        :param filters_num: number of filters in output tensor.
        :param stride: convolution stride.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.strides = stride
        self.ker_reg = None if regularization is None else regularization
        self.num_filters = num_filters

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'strides': self.strides,
            'ker_reg': self.ker_reg,
            'num_filters': self.num_filters,
        })
        return config

    def build(self, input_shape):
        self.BatchNorm = BatchNormalization(input_shape=input_shape)
        self.conv_1 = Conv2D(self.num_filters, (3, 3), strides=self.strides, activation=None,
                             padding='same', kernel_regularizer=self.ker_reg)
        self.conv_2 = Conv2D(self.num_filters, (3, 3), activation=None, padding='same', kernel_regularizer=self.ker_reg)
        self.conv_3 = Conv2D(self.num_filters, (1, 1), strides=self.strides, activation=None,
                             padding='same', kernel_regularizer=self.ker_reg)
        self.activation = LeakyReLU()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.BatchNorm(x)
        x = self.activation(x)
        x = self.conv_2(x)
        if self.strides == 1:
            x_1 = inputs
        else:
            x_1 = self.conv_3(inputs)
        x = Add()([x, x_1])
        x = self.BatchNorm(x)
        output = self.activation(x)
        return output


class DeconvolutionBlock(Layer):

    def __init__(self, num_filters, filter_size, strides, **kwargs):
        super(DeconvolutionBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'strides': self.strides,
            'filter_size': self.filter_size,
            'num_filters': self.num_filters,
        })
        return config

    def build(self, input_shape):
        self.deconv = tf.keras.layers.Conv2DTranspose(input_shape=input_shape, filters=self.num_filters,
                                                      kernel_size=self.filter_size, strides=self.strides,
                                                      padding='same')
        self.norm = BatchNormalization()
        self.activation = LeakyReLU(alpha=0.3)

    def call(self, inputs, training=False):
        x = self.deconv(inputs)
        x = self.norm(x, training=training)
        x = self.activation(x)
        return x


class CrossCorrelation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CrossCorrelation, self).__init__(**kwargs)

    def call(self, inputs):
        corr_filters, img = inputs[0], inputs[1]

        corr_filters = self.zero_mean(corr_filters)
        img = self.zero_mean(img)
        img = BatchNormalization()(img)
        correlation = tf.map_fn(
            fn=lambda t: tf.nn.conv2d(tf.expand_dims(t[0], 0), tf.expand_dims(t[1], 3),
                                      strides=[1, 1, 1, 1], padding="SAME"),
            elems=[img, corr_filters],
            fn_output_signature=tf.float32
        )
        correlation = tf.squeeze(correlation, axis=1)
        correlation = self.min_max_scaler(correlation)
        return correlation

    @staticmethod
    def min_max_scaler(inputs):
        inputs = (inputs - tf.reduce_min(inputs, axis=(1, 2), keepdims=True)) / (
                tf.reduce_max(inputs, axis=(1, 2), keepdims=True) - tf.reduce_min(inputs, axis=(1, 2), keepdims=True)
                + tf.keras.backend.epsilon()
        ) + tf.keras.backend.epsilon()
        return tf.math.abs(inputs)

    @staticmethod
    def zero_mean(inputs):
        return tf.math.subtract(inputs, tf.reduce_mean(inputs, axis=(1, 2), keepdims=True))


class NewCrossCorrelation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CrossCorrelation, self).__init__(**kwargs)

    def call(self, inputs):
        corr_filter, img = inputs[0], inputs[1]

        corr_filter = self.zero_mean(corr_filter)
        img = self.zero_mean(img)
        img = BatchNormalization()(img)
        correlation = tf.nn.conv2d(corr_filter, img, strides=[1, 1, 1, 1], padding="SAME")
        correlation = tf.squeeze(correlation, axis=1)
        correlation = self.min_max_scaler(correlation)
        return correlation

    @staticmethod
    def min_max_scaler(inputs):
        inputs = (inputs - tf.reduce_min(inputs, axis=(1, 2), keepdims=True)) / (
                tf.reduce_max(inputs, axis=(1, 2), keepdims=True) - tf.reduce_min(inputs, axis=(1, 2), keepdims=True)
                + tf.keras.backend.epsilon()
        ) + tf.keras.backend.epsilon()
        return tf.math.abs(inputs)

    @staticmethod
    def zero_mean(inputs):
        return tf.math.subtract(inputs, tf.reduce_mean(inputs, axis=(1, 2), keepdims=True))


class CustomModel(Model):

    def train_step(self, data):
        images, ground_truth, sample_weights = data
        gt_corr, y_true = ground_truth
        initial_filter_vector = tf.random.normal(shape=(1, 100))

        with tf.GradientTape() as tape:
            pred_correlation = self([initial_filter_vector, images], training=True)[0]
            loss = self.compiled_loss(gt_corr, pred_correlation, sample_weight=sample_weights,
                                      regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, pred_correlation)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, ground_truth, sample_weights = data
        gt_corr, y_true = ground_truth
        initial_filter_vector = tf.random.normal(shape=(1, 100))
        pred_correlation = self([initial_filter_vector, images], training=False)[0]
        self.compiled_loss(gt_corr, pred_correlation, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y_true, pred_correlation)
        return {m.name: m.result() for m in self.metrics}

    def get_correlation_filter(self):
        cf = self([tf.random.normal(shape=(1, 100)), tf.random.normal(shape=(1, 100, 100, 1))], training=False)[0]
        return cf[0, :, :, 0].numpy()

    def plot_output_correlations(self, data):
        input_data, ground_truth, _ = data
        initial_filter_matrix, images = input_data[0], input_data[1]
        correlations = self([initial_filter_matrix, images], training=False)[0]
        # labels = ground_truth[:, images.shape[1] // 2, images.shape[2] // 2, 0]
        # print(np.shape(ground_truth))
        labels = ground_truth[1]
        PlotCrossCorrelation(corr_scenes=correlations, labels=labels).plot()


class NNCFModel:
    def __init__(self, input_shape: [Tuple[int, int, int], Tuple[int, int, int]], num_classes: int, num_filters: int,
                 regularization: Optional[float], input_name: str):
        """
        Custom implementation of the stripped-down ResNet18 for cf synthesis task.
        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param input_name: name of the input tensor.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_name = input_name
        self.model_path = 'cfnn.h5'
        self.init_filters = num_filters
        self.ker_reg = None if regularization is None else tf.keras.regularizers.l2(regularization)

    def peak_classification(self):
        classifier = tf.keras.models.load_model(self.model_path)
        classifier.trainable = False
        return classifier

    def build(self) -> tf.keras.models.Model:
        """
        Building CNN model for cf synthesis task.
        :return: CustomModel() object.
        """
        inputs_1 = Input(shape=self.input_shape[0], name=self.input_name)

        x = Conv2D(self.init_filters, (7, 7), strides=2, kernel_regularizer=self.ker_reg)(inputs_1)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(2):
            x = ResidualBlock(num_filters=self.init_filters, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 2, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 2, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 4, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 4, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 8, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 8, regularization=self.ker_reg)(x)

        x = Conv2D(self.init_filters * 16, (1, 1), strides=1, kernel_regularizer=self.ker_reg)(x)

        x = DeconvolutionBlock(num_filters=self.init_filters * 8, filter_size=5, strides=2)(x)
        x = DeconvolutionBlock(num_filters=self.init_filters * 4, filter_size=5, strides=2)(x)
        x = DeconvolutionBlock(num_filters=self.init_filters * 2, filter_size=5, strides=2)(x)
        x = DeconvolutionBlock(num_filters=self.init_filters, filter_size=5, strides=2)(x)
        cf = Conv2D(1, (5, 5), activation='linear', padding='same', name='correlation_filter')(x)
        # x = Flatten()(x)
        # x = Dense(self.input_shape[1] ** 2, activation='linear', trainable=False)(x)
        # cf = Reshape(self.input_shape[1:], name='corr_filter')(x)

        inputs_2 = Input(shape=self.input_shape[1], name='images')
        x = CrossCorrelation()([cf, inputs_2])
        return Model(inputs=[inputs_1, inputs_2], outputs=[x, cf])


class GenerativeNNCFModel:
    def __init__(self, input_shape: [Tuple[int, int, int], Tuple[int, int, int]], num_filters: int,
                 regularization: Optional[float], input_name: str, nn_depth: int):
        """
        Custom implementation of the DC generator for cf synthesis task.
        :param input_shape: input shape (height, width, channels).
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param input_name: name of the input tensor.
        :param nn_depth: number of deconvolution layers.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        """
        self.init_filters = num_filters
        self.input_shape = input_shape
        self.input_name = input_name
        self.nn_depth = nn_depth
        self.start_size = (input_shape[1][0] // 2 ** nn_depth, input_shape[1][1] // 2 ** nn_depth,
                           num_filters * 2 ** nn_depth)
        self.model_path = 'gcfnn.h5'
        self.ker_reg = None if regularization is None else tf.keras.regularizers.l2(regularization)

    def build(self) -> tf.keras.models.Model:
        """
        Building CNN model for cf synthesis task.
        :return: CustomModel() object.
        """
        inputs_1 = Input(shape=(100,), name=self.input_name)

        x = Dense(self.start_size[0] * self.start_size[1] * self.start_size[2], use_bias=False)(inputs_1)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape(self.start_size)(x)

        for i in range(self.nn_depth - 1, -1, -1):
            x = DeconvolutionBlock(num_filters=self.init_filters * 2 ** i, filter_size=5, strides=2)(x)

        cf = Conv2D(1, (5, 5), activation='linear', padding='same', name='Correlation_Filter')(x)

        inputs_2 = Input(shape=self.input_shape[1], name='Train images')
        x = CrossCorrelation()([cf, inputs_2])
        return Model(inputs=[inputs_1, inputs_2], outputs=[x, cf])


tf.keras.backend.set_learning_phase(0)
a = 48
model = GenerativeNNCFModel(input_shape=[(1, 100), (a, a, 1)], num_filters=16, nn_depth=4,
                            regularization=0.0005, input_name='noise vector').build()
print(model.summary())
model.save('nncf_model1.h5')
