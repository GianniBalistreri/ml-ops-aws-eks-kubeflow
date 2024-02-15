"""

Convolutional Neural Network Classifier

"""

import datetime
import keras
import keras.backend as K
import numpy as np
import os
import tensorflow as tf

from custom_logger import Log
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal,
    RandomNormal, RandomUniform, TruncatedNormal, Zeros
)
from keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, PReLU, ReLU, Softmax
)
from keras.models import Model
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
from typing import List, Tuple


class ImageClassifierException(Exception):
    """
    Class for handling exceptions for class ImageClassifier
    """
    pass


class ImageClassifier:
    """
    Class for image classification
    """
    def __init__(self,
                 file_path_train_images: str,
                 file_path_test_images: str,
                 n_labels: int,
                 n_channels: int = 3,
                 image_height: int = 256,
                 image_width: int = 256,
                 learning_rate: float = 0.001,
                 n_epoch: int = 10,
                 optimizer: str = 'adam',
                 initializer: str = 'he_normal',
                 activation: str = 'relu',
                 batch_size: int = 1,
                 n_conv_layers: int = 7,
                 start_n_filters_conv_layers: int = 32,
                 max_n_filters_conv_layers: int = 256,
                 up_size_n_filters_period: int = 3,
                 dropout_rate_conv_layers: float = 0.0,
                 pool_size_conv_layers: Tuple[int, int] = (2, 2),
                 checkpoint_epoch_interval: int = 5,
                 print_model_architecture: bool = True,
                 **kwargs
                 ):
        """
        :param file_path_train_images: str
            File path of images for training

        :param file_path_test_images: str
            File path of images for testing

        :param n_labels: int
            Number of class labels

        :param n_channels: int
            Number of image channels
                -> 1: gray
                -> 3: color (rbg)

        :param image_height: int
            Height of the image

        :param image_width: int
            Width of the image

        :param learning_rate: float
            Learning rate

        :param initializer: str
            Name of the initializer used in convolutional layers
                -> constant: Constant value 2
                -> he_normal:
                -> he_uniform:
                -> glorot_normal: Xavier normal
                -> glorot_uniform: Xavier uniform
                -> lecun_normal: Lecun normal
                -> lecun_uniform:
                -> ones: Constant value 1
                -> orthogonal:
                -> random_normal:
                -> random_uniform:
                -> truncated_normal:
                -> zeros: Constant value 0

        :param activation: str
            Name of the activation function used in convolutional layers:
                -> leaky_relu: LeakyReLu
                -> prelu: PReLu
                -> relu: ReLu

        :param batch_size: int
            Batch size

        :param n_conv_layers: int
            Number of convolutional layers

        :param start_n_filters_conv_layers: int
            Number of filters used in first convolutional layer

        :param max_n_filters_discriminator: int
            Maximum number of filter used in all convolutional layers

        :param up_size_n_filters_period: int
            Period for up-sizing filters in convolutional layers

        :param dropout_rate_conv_layers: float
            Dropout rate used after each convolutional layer

        :param pool_size_conv_layers: int
            Pool size used in max pooling layer

        :param print_model_architecture: bool
            Whether to print architecture of cycle-gan model components (discriminators & generators) or not

        :param kwargs: dict
            Key-word arguments for class ImageProcessor and compiling model configuration
        """
        self.file_path_train_images: str = file_path_train_images
        self.file_path_test_images: str = file_path_test_images
        self.n_labels: int = n_labels
        self.n_epoch: int = n_epoch
        if 1 < n_channels < 4:
            self.n_channels: int = 3
        else:
            self.n_channels: int = 1
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.image_shape: tuple = tuple([self.image_width, self.image_height, self.n_channels])
        self.normalize: bool = False if kwargs.get('normalize') is None else kwargs.get('normalize')
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.001
        if optimizer == 'rmsprop':
            self.optimizer: RMSprop = RMSprop(learning_rate=self.learning_rate,
                                              rho=0.9,
                                              momentum=0.0,
                                              epsilon=1e-7,
                                              centered=False
                                              )
        elif optimizer == 'sgd':
            self.optimizer: SGD = SGD(learning_rate=self.learning_rate,
                                      momentum=0.0,
                                      nesterov=False
                                      )
        else:
            self.optimizer: Adam = Adam(learning_rate=self.learning_rate,
                                        beta_1=0.5,
                                        beta_2=0.999,
                                        epsilon=1e-7,
                                        amsgrad=False
                                        )
        self.batch_size: int = batch_size if batch_size > 0 else 1
        if self.batch_size == 1:
            self.normalizer = InstanceNormalization
        else:
            self.normalizer = BatchNormalization
        if initializer == 'constant':
            self.initializer: keras.initializers = Constant(value=2)
        elif initializer == 'he_normal':
            self.initializer: keras.initializers = HeNormal(seed=1234)
        elif initializer == 'he_uniform':
            self.initializer: keras.initializers = HeUniform(seed=1234)
        elif initializer == 'glorot_normal':
            self.initializer: keras.initializers = GlorotNormal(seed=1234)
        elif initializer == 'glorot_uniform':
            self.initializer: keras.initializers = GlorotUniform(seed=1234)
        elif initializer == 'lecun_normal':
            self.initializer: keras.initializers = LecunNormal(seed=1234)
        elif initializer == 'lecun_uniform':
            self.initializer: keras.initializers = LecunUniform(seed=1234)
        elif initializer == 'ones':
            self.initializer: keras.initializers = Ones()
        elif initializer == 'orthogonal':
            self.initializer: keras.initializers = Orthogonal(gain=1.0, seed=1234)
        elif initializer == 'random_normal':
            self.initializer: keras.initializers = RandomNormal(mean=0.0, stddev=0.2, seed=1234)
        elif initializer == 'random_uniform':
            self.initializer: keras.initializers = RandomUniform(minval=-0.05, maxval=0.05, seed=1234)
        elif initializer == 'truncated_normal':
            self.initializer: keras.initializers = TruncatedNormal(mean=0.0, stddev=0.05, seed=1234)
        elif initializer == 'zeros':
            self.initializer: keras.initializers = Zeros()
        if activation == 'leakly_relu':
            self.activation: keras.layers = LeakyReLU(alpha=0.3)
        elif activation == 'relu':
            self.activation: keras.layers = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
        elif activation == 'prelu':
            self.activation: keras.layers = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
        self.n_conv_layers: int = n_conv_layers
        self.start_n_filters_conv_layers: int = start_n_filters_conv_layers
        self.max_n_filters_conv_layers: int = max_n_filters_conv_layers
        self.up_size_n_filters_period: int = up_size_n_filters_period
        self.dropout_rate_conv_layers: float = dropout_rate_conv_layers
        self.pool_size_conv_layers: Tuple[int, int] = pool_size_conv_layers
        self.print_model_architecture: bool = print_model_architecture
        self.kwargs: dict = kwargs
        self.model_name: str = None
        self.model: Model = None
        self.training_time: str = None
        self.elapsed_time: List[str] = []
        self.epoch: List[int] = []
        self.batch: List[int] = []
        self.training_report: dict = {}
        self.n_gpu: int = 0

    def _build_classifier(self) -> Model:
        """
        Build classification model network

        :return: Model:
            Neural network
        """
        _input: Input = Input(shape=self.image_shape)
        _n_filters: int = self.start_n_filters_conv_layers
        _c: tf.Tensor = Conv2D(filters=self.start_n_filters_conv_layers,
                               kernel_size=(2, 2),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=self.initializer
                               )(_input)
        _c = self.activation(_c)
        if self.dropout_rate_conv_layers > 0:
            _c = Dropout(rate=self.dropout_rate_conv_layers)(_c)
        _c = MaxPooling2D(pool_size=self.pool_size_conv_layers,
                          strides=(1, 1),
                          padding='valid'
                          )
        for c in range(0, self.n_conv_layers - 1, 1):
            if c % self.up_size_n_filters_period == 0:
                _n_filters *= 2
            _c: tf.Tensor = Conv2D(filters=_n_filters,
                                   kernel_size=(2, 2),
                                   strides=(1, 1),
                                   padding='valid',
                                   kernel_initializer=self.initializer
                                   )(_input)
            _c = self.activation(_c)
            if self.dropout_rate_conv_layers > 0:
                _c = Dropout(rate=self.dropout_rate_conv_layers)(_c)
            _c = MaxPooling2D(pool_size=self.pool_size_conv_layers,
                              strides=(1, 1),
                              padding='valid'
                              )(_c)
        _c = Flatten()(_c)
        _c = Dense(units=_n_filters,
                   use_bias=True,
                   kernel_initializer=self.initializer,
                   )(_c)
        _c = self.activation(_c)
        if self.dropout_rate_conv_layers > 0:
            _c = Dropout(rate=self.dropout_rate_conv_layers)(_c)
        _c = Dense(units=self.n_labels,
                   use_bias=True,
                   kernel_initializer=self.initializer
                   )(_c)
        _output: tf.Tensor = Softmax(axis=-1)(_c)
        return Model(inputs=_input, outputs=_output, name=self.model_name)

    def _train(self) -> None:
        """
        Compile and fit neural network model
        """
        # Load and preprocess image data (train & validation):
        _train_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255,
                                                                       shear_range=0.0,
                                                                       zoom_range=0.0,
                                                                       horizontal_flip=True
                                                                       )
        _validation_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255)
        _train_generator = _train_data_generator.flow_from_directory(directory=self.file_path_train_images,
                                                                     target_size=(self.image_width, self.image_height),
                                                                     batch_size=self.batch_size,
                                                                     class_mode='categorical'
                                                                     )
        _validation_generator = _validation_data_generator.flow_from_directory(directory=self.file_path_test_images,
                                                                               target_size=(self.image_width, self.image_height),
                                                                               batch_size=self.batch_size,
                                                                               class_mode='categorical'
                                                                               )
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=[tf.keras.metrics.Accuracy(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.TruePositives(),
                                    tf.keras.metrics.FalsePositives(),
                                    tf.keras.metrics.TrueNegatives(),
                                    tf.keras.metrics.FalseNegatives()
                                    ]
                           )
        # Train neural network model:
        self.model.fit(x=_train_generator, epochs=self.n_epoch, validation_data=_validation_generator)

    def inference(self, image: np.ndarray) -> int:
        """
        Predict image class

        :param image: np.ndarray
            Image

        :return: int
            Class label number
        """
        return self.model.predict(x=image)

    def train(self) -> None:
        """
        Train image classifier
        """
        Log().log(msg=f'Physical GPU devices: {tf.config.list_physical_devices("GPU")}')
        _strategy = tf.distribute.MirroredStrategy()
        self.n_gpu = _strategy.num_replicas_in_sync
        Log().log(msg=f'Number of devices to use: {self.n_gpu}')
        if self.n_gpu > 1:
            with _strategy.scope():
                self.model = self._build_classifier()
                self._train()
        else:
            self.model = self._build_classifier()
            self._train()
