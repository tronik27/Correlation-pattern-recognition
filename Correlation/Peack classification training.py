import tensorflow as tf
from Peak_classification import CustomResNet18, CorrelationGenerator
from nncf_utils import make_tensorboard
from Correlation_utils import PlotCrossCorrelation
import numpy as np


def get_metrics():
    metrics = list()
    metrics.append(tf.keras.metrics.BinaryAccuracy(threshold=0.5))
    metrics.append(tf.keras.metrics.Precision())
    metrics.append(tf.keras.metrics.Recall())
    return metrics


def get_callbacks():
    callbacks = []

    tensorboard_callback = make_tensorboard()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.0001,
                                                     factor=0.1, patience=3, min_lr=0.00001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('training_pcnn/weights', save_weights_only=True,
                                                    save_best_only=True, monitor='val_loss', mode='min')
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.00001)

    callbacks += [tensorboard_callback, reduce_lr, checkpoint, stop]
    return callbacks


def train(images_path, labels_path=None, validation_path=None, validation_labels_path=None,  num_of_correlations=12000,
          batch_size=32, epochs=1000, target_size=(100, 100), show_learning_curves=False, learning_rate=0.001,
          inverse=False):
    train_data = CorrelationGenerator(images_path=images_path,
                                      num_of_correlations=num_of_correlations,
                                      labels_path=labels_path,
                                      input_size=target_size,
                                      inverse=inverse,
                                      batch_size=batch_size).get_generator()
    valid_data = CorrelationGenerator(images_path=validation_path,
                                      num_of_correlations=num_of_correlations // 4,
                                      labels_path=validation_labels_path,
                                      input_size=target_size,
                                      inverse=inverse,
                                      batch_size=batch_size).get_generator()

    pcnn = CustomResNet18(input_shape=(target_size[0], target_size[1], 1), num_classes=2, alpha=1.,
                          regularization=0.0005, activation_type='relu', input_name='cpr').build()

    print(pcnn.summary())

    pcnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=get_metrics())

    history = pcnn.fit(train_data, validation_data=valid_data, steps_per_epoch=num_of_correlations // batch_size,
                       callbacks=get_callbacks(), epochs=epochs)

    best_pcnn = CustomResNet18(input_shape=(target_size[0], target_size[1], 1), num_classes=2, alpha=1.,
                               regularization=0.0005, activation_type='relu', input_name='cpr').build()
    best_pcnn.load_weights('training_pcnn/weights')
    best_pcnn.save('pcnn.h5')


def test(images_path, labels_path=None, num_of_correlations=12000, target_size=(100, 100), inverse=False):
    test_data = CorrelationGenerator(images_path=images_path,
                                     num_of_correlations=num_of_correlations,
                                     labels_path=labels_path,
                                     input_size=target_size,
                                     inverse=inverse,
                                     batch_size=32).get_generator()
    classifier = tf.keras.models.load_model('pcnn.h5')
    x, y = test_data[0]
    PlotCrossCorrelation(corr_scenes=x, labels=np.argmax(y, axis=1)).plot()
    predicted_labels = classifier(x, training=False)
    print(np.argmax(y, axis=1))
    print(np.argmax(predicted_labels, axis=1))
    print('Accuracy:', tf.reduce_sum(tf.keras.metrics.binary_accuracy(y, predicted_labels)).numpy() /
          num_of_correlations)


train(images_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train',
      labels_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv',
      validation_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test',
      validation_labels_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv',
      num_of_correlations=10000, batch_size=32, epochs=1000, target_size=(48, 48), learning_rate=0.001, inverse=False)
# test(images_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test',
#      labels_path='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv',
#      num_of_correlations=32, target_size=(48, 48), inverse=False)
