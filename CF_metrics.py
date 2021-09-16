import numpy as np
import sklearn
from Correlation.Correlation_utils import PlotCrossCorrelation
import tensorflow as tf
from onnxruntime import InferenceSession


class OnnxModelLoader:
    def __init__(self, onnx_path: str):
        """
        Class for loading ONNX models to inference on CPU.
        :param onnx_path: path to ONNX model file (*.onnx file).
        """
        self.onnx_path = onnx_path
        self.sess = InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

        # In current case model always has exactly one input and one output.
        self.input_name = [x.name for x in self.sess.get_inputs()][0]
        self.output_name = [x.name for x in self.sess.get_outputs()][0]

    def inference(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference.
        :param inputs: input numpy array.
        :return: output numpy array.
        """
        outputs = self.sess.run([self.output_name], input_feed={self.input_name: np.float32(inputs)})
        return outputs[0]


class CFMetric(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy', peak_classifier='pcnn', path_to_model='frozen_model.ONNX'):
        super(CFMetric, self).__init__(name=name)
        metrics = {'f1': sklearn.metrics.f1_score,
                   'precision': sklearn.metrics.precision_score,
                   'recall': sklearn.metrics.recall_score,
                   'accuracy': sklearn.metrics.accuracy_score}
        if peak_classifier == 'peak_position':
            self.peak_classifier = self.peak_position
        elif peak_classifier == 'pcnn':
            self.model = OnnxModelLoader(path_to_model)
            self.peak_classifier = self.pcnn
        self.metric_value = self.add_weight(name=name, initializer='zeros')
        self.metric = metrics[name]
        self.path_to_model = path_to_model

    def update_state(self, y_true, correlation, sample_weight=None):
        if len(correlation.shape) not in (2, 3, 4):
            raise ValueError('Incorrect shape of correlation!')
        elif len(correlation.shape) == 2:
            correlation = np.expand_dims(correlation, axis=(0, 3))
        elif len(correlation.shape) == 3:
            correlation = np.expand_dims(correlation, axis=3)

        if len(y_true.shape) > 1:
            raise ValueError('Incorrect shape of labels array! It should be 1 - dimensional!')
        assert correlation.shape[0] == y_true.shape[0]

        if correlation.shape[1] > 32:
            correlation = correlation[:, correlation.shape[1] // 2 - 16: correlation.shape[1] // 2 + 16,
                                      correlation.shape[2] // 2 - 16: correlation.shape[2] // 2 + 16, :]

        y_pred = self.peak_classifier(correlation)
        self.metric_value.assign_add(tf.reduce_sum(tf.cast(self.metric(y_true, y_pred), self.dtype)))

    def result(self):
        return self.metric_value

    def pcnn(self, scenes):
        class_labels = []
        for i in range(scenes.shape[0]):
            scene = tf.expand_dims(scenes[i, :, :, :] / tf.reduce_max(scenes[i, :, :, :], axis=[0, 1], keepdims=True),
                                   axis=0)
            predicted_labels = self.model.inference(scene)
            if predicted_labels[0, 0] >= 0.5:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return np.array(class_labels)

    @staticmethod
    def peak_position(scenes):
        if scenes.shape[1] % 2:
            x1, x2 = scenes.shape[1] // 2 - 1, scenes.shape[1] // 2 + 2
        else:
            x1, x2 = scenes.shape[1] // 2 - 1, scenes.shape[1] // 2 + 1
        class_labels = []
        for scene in scenes[:, :, :, 0]:
            peak_coordinates = np.unravel_index(scene.argmax(), scene.shape)
            if x1 <= peak_coordinates[0] <= x2 and x1 <= peak_coordinates[1] <= x2:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return np.array(class_labels)


class PeakMetric:

    def __init__(self, correlation):
        if len(correlation.shape) not in (2, 3, 4):
            raise ValueError('Incorrect shape of correlation!')
        elif len(correlation.shape) == 2:
            correlation = np.expand_dims(correlation, axis=(0, 3))
        elif len(correlation.shape) == 3:
            correlation = np.expand_dims(correlation, axis=3)
        self.scenes = correlation[:, correlation.shape[1] // 2 - 16: correlation.shape[1] // 2 + 16,
                                  correlation.shape[2] // 2 - 16: correlation.shape[2] // 2 + 16, :]

    def peak_height(self):
        return tf.reduce_max(self.scenes, axis=[1, 2])

    def psr(self):
        psr = (tf.reduce_max(self.scenes, axis=[1, 2], keepdims=True)
               - tf.reduce_mean(self.scenes, axis=[1, 2], keepdims=True)) \
              / tf.math.reduce_std(self.scenes, axis=[1, 2], keepdims=True)
        return psr

    def pce(self):
        pse = tf.reduce_max(self.scenes, axis=[1, 2], keepdims=True) / \
              tf.math.reduce_sum(self.scenes, axis=[1, 2], keepdims=True)
        return pse
