from tensorflow.keras.layers import Reshape, Layer, BatchNormalization, Conv2D, Dense, Lambda, Add, \
    GlobalAveragePooling2D, UpSampling2D, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from Correlation_utils import PlotCrossCorrelation
import tensorflow_addons as tfa
import numpy as np
import keras2onnx


class ResidualBlock(Layer):

    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.BatchNorm_1 = BatchNormalization(input_shape=input_shape[1:2])
        self.conv_1 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')
        self.BatchNorm_2 = BatchNormalization()
        self.conv_2 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')

    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        x = self.BatchNorm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.BatchNorm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        return Add()([x, inputs])


class FiltersChangeResidualBlock(Layer):

    def __init__(self, out_filters, **kwargs):
        """
        The class initialiser should call the base class initialiser, passing any keyword
        arguments along. It should also set the number of filters as a class attribute.
        """
        super(FiltersChangeResidualBlock, self).__init__(**kwargs)
        self.out_filters = out_filters

    def build(self, input_shape):
        """
        This method should build the layers according to the above specification. Make sure
        to use the input_shape argument to get the correct number of filters, and to set the
        input_shape of the first layer in the block.
        """
        self.BatchNorm_1 = BatchNormalization(input_shape=input_shape[1:2])
        self.conv_1 = Conv2D(input_shape[-1], (3, 3), activation=None, padding='same')
        self.BatchNorm_2 = BatchNormalization()
        self.conv_2 = Conv2D(self.out_filters, (3, 3), activation=None, padding='same')
        self.conv_3 = Conv2D(self.out_filters, (1, 1), activation=None)

    def call(self, inputs, training=False):
        """
        This method should contain the code for calling the layer according to the above
        specification, using the layer objects set up in the build method.
        """
        x = self.BatchNorm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.BatchNorm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x1 = self.conv_3(inputs)
        return Add()([x, x1])


class DeconvolutionBlock(Layer):

    def __init__(self, num_filters, filter_size, strides, **kwargs):
        super(DeconvolutionBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides

    def build(self, input_shape):
        self.deconv = tf.keras.layers.Conv2DTranspose(input_shape=input_shape, filters=self.num_filters,
                                                      kernel_size=self.filter_size, strides=self.strides,
                                                      padding='same')
        # self.Norm = BatchNormalization()
        self.norm = tfa.layers.InstanceNormalization(gamma_initializer='random_uniform')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.3)

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
        correlation = tf.map_fn(
            fn=lambda t: tf.nn.conv2d(tf.expand_dims(t[0], 0), tf.expand_dims(t[1], 3),
                                      strides=[1, 1, 1, 1], padding="SAME"),
            elems=[img, corr_filters],
            fn_output_signature=tf.float32
        )
        correlation = tf.squeeze(correlation, axis=1)
        # correlation = tf.math.abs(correlation) + tf.keras.backend.epsilon()
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


class NNCFModel(Model):

    def __init__(self, num_correlations=32):
        super(NNCFModel, self).__init__()
        self.num_correlations = num_correlations
        # self.shape = shape
        self.conv_1 = Conv2D(32, (7, 7), activation='relu', strides=(2, 2), padding='same')
        self.residual_block = ResidualBlock()
        self.conv_2 = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')
        self.filters_change_residual_block = FiltersChangeResidualBlock(64)
        self.conv_3 = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')
        self.deconv_1 = DeconvolutionBlock(num_filters=32, filter_size=5, strides=2)
        self.deconv_2 = DeconvolutionBlock(num_filters=16, filter_size=5, strides=2)
        self.deconv_3 = DeconvolutionBlock(num_filters=8, filter_size=5, strides=2)
        self.conv_4 = Conv2D(1, (5, 5), activation='sigmoid', padding='same')
        # self.conv_3 = Conv2D(self.shape[1] ** 2, (1, 1), activation='sigmoid', strides=(1, 1))
        # self.average_pooling = GlobalAveragePooling2D()
        # self.reshape = Reshape(self.shape[1:], name='corr_filter')
        self.correlation = CrossCorrelation()

    def call(self, inputs):
        x = self.conv_1(inputs[0])
        x = self.residual_block(x)
        x = self.conv_2(x)
        x = self.filters_change_residual_block(x)
        x = self.conv_3(x)
        # x = self.average_pooling(x)
        # cf = self.reshape(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        cf = self.conv_4(x)

        x = self.correlation([cf, inputs[1]])
        return x, cf

    def train_step(self, data):
        input_data, ground_truth, sample_weights = data
        initial_filter_matrix, images = input_data[0], input_data[1]

        with tf.GradientTape() as tape:
            y_pred = self([initial_filter_matrix, images], training=True)[0]
            loss = self.compiled_loss(ground_truth, y_pred, sample_weight=sample_weights,
                                      regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(ground_truth, y_pred, sample_weight=sample_weights)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        input_data, ground_truth, sample_weights = data
        initial_filter_matrix, images = input_data[0], input_data[1]
        y_pred = self([initial_filter_matrix, images], training=False)[0]
        self.compiled_loss(ground_truth, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(ground_truth, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_correlation_filter(self, data):
        input_data, _, _ = data
        initial_filter_matrix, images = input_data[0], input_data[1]
        cf = self([initial_filter_matrix, images], training=False)[1]
        return cf[0, :, :, 0].numpy()

    def plot_output_correlations(self, data):
        input_data, ground_truth, _ = data
        initial_filter_matrix, images = input_data[0], input_data[1]
        correlations = self([initial_filter_matrix, images], training=False)[0]
        labels = ground_truth[:, images.shape[1] // 2, images.shape[2] // 2, 0]
        print(ground_truth.shape)
        PlotCrossCorrelation(corr_scenes=correlations, labels=labels).plot()

    def model(self, shape):
        x1 = tf.keras.layers.Input(shape=shape)
        x2 = tf.keras.layers.Input(shape=shape)
        return Model(inputs=[x1, x2], outputs=self.call([x1, x2]))


# tf.keras.backend.set_learning_phase(0)
# nncf = NNCFModel()
# _ = nncf([tf.random.uniform((32, 128, 128, 1)), tf.random.uniform((32, 128, 128, 1))])
# keras2onnx.convert_keras(nncf, nncf.name)
#
# print(nncf.summary())
# keras2onnx.save_model(nncf, 'nncf.onnx')
