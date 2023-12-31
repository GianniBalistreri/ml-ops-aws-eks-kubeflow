"""

Cycle GAN (Generative Adversarial Network)

"""

import datetime
import json
import keras
import keras.backend as K
import numpy as np
import os

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from deep_learning_layers import ReflectionPadding2D
from image_processor import ImageProcessor
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine import keras_tensor
from keras.initializers.initializers_v2 import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, Zeros
)
from keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Dense, Dropout, Flatten, Input, Multiply, ReLU, ZeroPadding2D
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import List


class CycleGANException(Exception):
    """
    Class for handling exceptions for class CycleGAN
    """
    pass


class CycleGAN:
    """
    Class for building and training cycle-gan model
    """
    def __init__(self,
                 file_path_train_clean_images: str,
                 file_path_train_noisy_images: str,
                 file_path_eval_noisy_images: str = None,
                 file_path_moe_noisy_images: List[str] = None,
                 n_channels: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 learning_rate: float = 0.0002,
                 optimizer: str = 'adam',
                 initializer: str = 'he_normal',
                 batch_size: int = 1,
                 start_n_filters_discriminator: int = 64,
                 max_n_filters_discriminator: int = 512,
                 n_conv_layers_discriminator: int = 3,
                 dropout_rate_discriminator: float = 0.0,
                 start_n_filters_generator: int = 32,
                 max_n_filters_generator: int = 512,
                 up_sample_n_filters_period: int = 0,
                 generator_type: str = 'res',
                 n_res_net_blocks: int = 6,
                 n_conv_layers_generator_res_net: int = 2,
                 n_conv_layers_generator_u_net: int = 3,
                 dropout_rate_generator_res_net: float = 0.0,
                 dropout_rate_generator_down_sampling: float = 0.0,
                 dropout_rate_generator_up_sampling: float = 0.0,
                 print_model_architecture: bool = True,
                 **kwargs
                 ):
        """
        :param file_path_train_clean_images: str
            Complete file path of clean images for training

        :param file_path_train_noisy_images: str
            Complete file path of noisy images for training

        :param file_path_eval_noisy_images: str
            Complete file path of noisy images for testing

        :param file_path_moe_noisy_images: List[str]
            Complete file paths of several noisy images

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

        :param batch_size: int
            Batch size

        :param start_n_filters_discriminator: int
            Number of filters used in first convolutional layer in discriminator network

        :param max_n_filters_discriminator: int
            Maximum number of filter used in all convolutional layers in discriminator network

        :param n_conv_layers_discriminator: int
            Number of convolutional layers in discriminator network

        :param dropout_rate_discriminator: float
            Dropout rate used after each convolutional layer in discriminator network

        :param start_n_filters_generator: int
            Number of filters used in first convolutional layer in generator network

        :param max_n_filters_generator: int
            Maximum number of filter used in all convolutional layers in generator network

        :param up_sample_n_filters_period: int
            Number of layers until up-sampling number of filters

        :param generator_type: str
            Abbreviated name of the type of the generator
                -> u: U-Network architecture
                -> resnet: Residual network architecture

        :param n_res_net_blocks: int
            Number of residual network blocks to use
                Common: -> 6, 9

        :param n_conv_layers_generator_res_net: int
            Number of convolutional layers used for down and up sampling

        :param n_conv_layers_generator_u_net: int
            Number of convolutional layers in generator network with u-net architecture

        :param dropout_rate_generator_res_net: float
            Dropout rate used after each convolutional layer in generator residual network

        :param dropout_rate_generator_down_sampling: float
            Dropout rate used after each convolutional layer in generator down-sampling network

        :param dropout_rate_generator_up_sampling: float
            Dropout rate used after each convolutional layer in generator up-sampling network

        :param print_model_architecture: bool
            Whether to print architecture of cycle-gan model components (discriminators & generators) or not

        :param kwargs: dict
            Key-word arguments for class ImageProcessor and compiling model configuration
        """
        if len(file_path_train_clean_images) == 0:
            raise CycleGANException('File path for clean training document images is empty')
        if len(file_path_train_noisy_images) == 0:
            raise CycleGANException('File path for noisy training document images is empty')
        self.file_path_train_clean_data: str = file_path_train_clean_images
        self.file_path_train_noisy_data: str = file_path_train_noisy_images
        self.file_path_eval_noisy_data: str = file_path_eval_noisy_images
        self.file_path_moe_noisy_images: List[str] = file_path_moe_noisy_images
        self.model_name_generator_A: str = 'generator_A'
        self.model_name_generator_B: str = 'generator_B'
        self.model_name_discriminator_A: str = 'discriminator_A'
        self.model_name_discriminator_B: str = 'discriminator_B'
        self.model_name_combined_model: str = 'combined_model'
        self.model_name_embedder: str = 'embedder'
        self.model_name_gated_network: str = 'gated_network'
        if 1 < n_channels < 4:
            self.n_channels: int = 3
        else:
            self.n_channels: int = 1
        self.image_height: int = image_height if image_height > 0 else 256
        self.image_width: int = image_width if image_width > 0 else 256
        self.image_shape: tuple = tuple([self.image_width, self.image_height, self.n_channels])
        self.normalize: bool = False if kwargs.get('normalize') is None else kwargs.get('normalize')
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.0002
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
        self.discriminator_batch_size: int = self.batch_size
        self.generator_batch_size: int = self.batch_size
        if self.batch_size == 1:
            self.normalizer = InstanceNormalization
        else:
            self.normalizer = BatchNormalization
        if initializer == 'constant':
            self.initializer: keras.initializers.initializers_v2 = Constant(value=2)
        elif initializer == 'he_normal':
            self.initializer: keras.initializers.initializers_v2 = HeNormal(seed=1234)
        elif initializer == 'he_uniform':
            self.initializer: keras.initializers.initializers_v2 = HeUniform(seed=1234)
        elif initializer == 'glorot_normal':
            self.initializer: keras.initializers.initializers_v2 = GlorotNormal(seed=1234)
        elif initializer == 'glorot_uniform':
            self.initializer: keras.initializers.initializers_v2 = GlorotUniform(seed=1234)
        elif initializer == 'lecun_normal':
            self.initializer: keras.initializers.initializers_v2 = LecunNormal(seed=1234)
        elif initializer == 'lecun_uniform':
            self.initializer: keras.initializers.initializers_v2 = LecunUniform(seed=1234)
        elif initializer == 'ones':
            self.initializer: keras.initializers.initializers_v2 = Ones()
        elif initializer == 'orthogonal':
            self.initializer: keras.initializers.initializers_v2 = Orthogonal(gain=1.0, seed=1234)
        elif initializer == 'random_normal':
            self.initializer: keras.initializers.initializers_v2 = RandomNormal(mean=0.0, stddev=0.2, seed=1234)
        elif initializer == 'random_uniform':
            self.initializer: keras.initializers.initializers_v2 = RandomUniform(minval=-0.05, maxval=0.05, seed=1234)
        elif initializer == 'truncated_normal':
            self.initializer: keras.initializers.initializers_v2 = TruncatedNormal(mean=0.0, stddev=0.05, seed=1234)
        elif initializer == 'zeros':
            self.initializer: keras.initializers.initializers_v2 = Zeros()
        self.start_n_filters_discriminator: int = start_n_filters_discriminator if start_n_filters_discriminator > 0 else 64
        self.max_n_filters_discriminator: int = max_n_filters_discriminator if max_n_filters_discriminator > 0 else 512
        self.n_conv_layers_discriminator: int = n_conv_layers_discriminator if n_conv_layers_discriminator > 0 else 3
        self.dropout_rate_discriminator: float = dropout_rate_discriminator if dropout_rate_discriminator >= 0 else 0.0
        self.start_n_filters_generator: int = start_n_filters_generator if start_n_filters_generator > 0 else 32
        self.max_n_filters_generator: int = max_n_filters_generator if max_n_filters_generator > 0 else 512
        self.up_sample_n_filters_period: int = up_sample_n_filters_period
        if generator_type in ['u', 'res']:
            self.generator_type: str = generator_type
        else:
            self.generator_type: str = 'u'
        self.n_conv_layers_generator_u_net: int = n_conv_layers_generator_u_net if n_conv_layers_generator_u_net > 0 else 3
        self.n_conv_layers_generator_res_net: int = n_conv_layers_generator_res_net if n_conv_layers_generator_res_net > 0 else 2
        self.n_res_net_blocks: int = n_res_net_blocks if n_res_net_blocks > 0 else 9
        self.dropout_rate_generator_res_net: float = dropout_rate_generator_res_net if dropout_rate_generator_res_net > 0 else 0.0
        self.dropout_rate_generator_down_sampling: float = dropout_rate_generator_down_sampling if dropout_rate_generator_down_sampling > 0 else 0.0
        self.dropout_rate_generator_up_sampling: float = dropout_rate_generator_up_sampling if dropout_rate_generator_up_sampling > 0 else 0.0
        self.print_model_architecture: bool = print_model_architecture
        self.kwargs: dict = kwargs
        self.clf_label: int = None
        self.training_type: str = None
        self.discriminator_patch: tuple = None
        self.discriminator_A: Model = None
        self.discriminator_B: Model = None
        self.generator_A: Model = None
        self.generator_B: Model = None
        self.combined_model: Model = None
        self.model_name: str = None
        self.training_time: str = None
        self.elapsed_time: List[str] = []
        self.epoch: List[int] = []
        self.batch: List[int] = []
        self.label: List[int] = []
        self.discriminator_loss: List[float] = []
        self.discriminator_accuracy: List[float] = []
        self.generator_loss: List[float] = []
        self.adversarial_loss: List[float] = []
        self.reconstruction_loss: List[float] = []
        self.identy_loss: List[float] = []
        # Cycle-consistency loss:
        self.lambda_cycle: float = 10.0
        # Identity loss:
        self.lambda_id: float = 0.1 * self.lambda_cycle
        # Initialize ImageProcessor for loading and preprocessing image data (clean & noisy) in training:
        self.image_processor: ImageProcessor = ImageProcessor(file_path_clean_images=self.file_path_train_clean_data,
                                                              file_path_noisy_images=self.file_path_train_noisy_data,
                                                              file_path_multi_noisy_images=self.file_path_moe_noisy_images,
                                                              n_channels=self.n_channels,
                                                              batch_size=self.batch_size,
                                                              image_resolution=(self.image_width, self.image_height),
                                                              normalize=self.normalize,
                                                              flip=True if self.kwargs.get('flip') is None else self.kwargs.get('flip'),
                                                              crop=self.kwargs.get('crop')
                                                              )
        self.training_report: dict = {}

    def _build_discriminator(self) -> Model:
        """
        Build discriminator network

        :return: Model
            Discriminator network model
        """
        _input: Input = Input(shape=(self.image_width, self.image_height, self.n_channels))
        _n_filters: int = self.start_n_filters_discriminator
        _d: keras_tensor.KerasTensor = Conv2D(filters=_n_filters,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              padding='same',
                                              kernel_initializer=self.initializer
                                              )(_input)
        _d = LeakyReLU(alpha=0.2)(_d)
        for _ in range(0, self.n_conv_layers_discriminator, 1):
            if _n_filters < self.max_n_filters_discriminator:
                _n_filters *= 2
            _d = self._convolutional_layer_discriminator(input_layer=_d, n_filters=_n_filters)
        _d = ZeroPadding2D(padding=(1, 1))(_d)
        _d = Conv2D(filters=_n_filters * 2,
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer=self.initializer
                    )(_d)
        _d = self.normalizer()(_d)
        _d = LeakyReLU(alpha=0.2)(_d)
        _d = ZeroPadding2D(padding=(1, 1))(_d)
        _patch_out: keras_tensor.KerasTensor = Conv2D(filters=self.n_channels,
                                                      kernel_size=(4, 4),
                                                      strides=(1, 1),
                                                      padding='valid',
                                                      kernel_initializer=self.initializer,
                                                      #activation='sigmoid'
                                                      )(_d)
        self.discriminator_patch = (K.int_shape(_patch_out[0])[0], K.int_shape(_patch_out[0])[1], K.int_shape(_patch_out[0])[2])
        return Model(inputs=_input, outputs=_patch_out, name=self.model_name)

    def _build_cycle_gan_network(self):
        """
        Build complete cycle-gan network
        """
        # Input images from both domains:
        _image_A: Input = Input(shape=self.image_shape)
        _image_B: Input = Input(shape=self.image_shape)
        # Build and compile the discriminators:
        self.model_name = 'discriminator_A'
        self.discriminator_A = self._build_discriminator()
        self.model_name = 'discriminator_B'
        self.discriminator_B = self._build_discriminator()
        if self.print_model_architecture:
            self.discriminator_A.summary()
        self.discriminator_A.compile(loss='mse' if self.kwargs.get('discriminator_loss') is None else self.kwargs.get('discriminator_loss'),
                                     optimizer=self.optimizer,
                                     metrics=['accuracy'] if self.kwargs.get('discriminator_metrics') is None else self.kwargs.get('discriminator_metrics'),
                                     loss_weights=[0.5] if self.kwargs.get('discriminator_loss_weights') is None else self.kwargs.get('discriminator_loss_weights')
                                     )
        self.discriminator_B.compile(loss='mse' if self.kwargs.get('discriminator_loss') is None else self.kwargs.get('discriminator_loss'),
                                     optimizer=self.optimizer,
                                     metrics=['accuracy'] if self.kwargs.get('discriminator_metrics') is None else self.kwargs.get('discriminator_metrics'),
                                     loss_weights=[0.5] if self.kwargs.get('discriminator_loss_weights') is None else self.kwargs.get('discriminator_loss_weights')
                                     )
        # For the combined model we will only train the generators:
        self.discriminator_A.trainable = False if self.kwargs.get('discriminator_trainable') is None else self.kwargs.get('discriminator_trainable')
        self.discriminator_B.trainable = False if self.kwargs.get('discriminator_trainable') is None else self.kwargs.get('discriminator_trainable')
        # Build the generators:
        self.model_name = 'generator_A'
        self.generator_A = self._build_generator()
        self.model_name = 'generator_B'
        self.generator_B = self._build_generator()
        if self.print_model_architecture:
            self.generator_A.summary()
        # Translate images to the other domain:
        _fake_B = self.generator_A(_image_B)
        _fake_A = self.generator_B(_image_A)
        # Translate images back to original domain:
        _reconstruction_A = self.generator_B(_fake_B)
        _reconstruction_B = self.generator_A(_fake_A)
        # Identity mapping of images:
        _image_id_A = self.generator_B(_image_A)
        _image_id_B = self.generator_A(_image_B)
        # Discriminators determine validity of translated images:
        _valid_A = self.discriminator_A(_fake_A)
        _valid_B = self.discriminator_B(_fake_B)
        # Combined model to train generators to fool discriminators:
        self.combined_model = Model(inputs=[_image_A,
                                            _image_B
                                            ],
                                    outputs=[_valid_A,
                                             _valid_B,
                                             _reconstruction_A,
                                             _reconstruction_B,
                                             _image_id_A,
                                             _image_id_B
                                             ]
                                    )
        self.combined_model.compile(loss=['mse',
                                          'mse',
                                          'mae',
                                          'mae',
                                          'mae',
                                          'mae'
                                          ],
                                    loss_weights=[1,
                                                  1,
                                                  self.lambda_cycle,
                                                  self.lambda_cycle,
                                                  self.lambda_id,
                                                  self.lambda_id
                                                  ],
                                    optimizer=self.optimizer
                                    )

    def _build_generator(self) -> Model:
        """
        Build generator network

        :return: Model
            Generator network model
        """
        if self.generator_type == 'u':
            # U-Network:
            return self._u_network()
        elif self.generator_type == 'res':
            # Residual Network:
            return self._residual_network()

    def _convolutional_layer_discriminator(self, input_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Convolutional layer for discriminator

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _d: keras_tensor.KerasTensor = Conv2D(filters=n_filters,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              padding='same',
                                              kernel_initializer=self.initializer
                                              )(input_layer)
        # _d = self.normalizer()(_d)
        _d = LeakyReLU(alpha=0.2)(_d)
        if self.dropout_rate_discriminator > 0:
            _d = Dropout(self.dropout_rate_discriminator)(_d)
        return _d

    def _convolutional_layer_generator_decoder(self,
                                               input_layer: keras_tensor.KerasTensor,
                                               skip_layer: keras_tensor.KerasTensor,
                                               n_filters: int
                                               ) -> keras_tensor.KerasTensor:
        """
        Convolutional layer decoder for up-sampling

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param skip_layer:
            Network layer to concatenate with up-sampling output

        :param n_filters: int
            Number of filters in the (transposed) convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        if self.generator_type == 'u':
            _u: keras_tensor.KerasTensor = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_layer)
            _u = Conv2D(filters=n_filters,
                        kernel_size=(4, 4),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.initializer
                        )(_u)
            #_u = Conv2DTranspose(filters=n_filters,
            #                     kernel_size=(4, 4),
            #                     strides=(2, 2),
            #                     padding='same',
            #                     kernel_initializer=self.initializer
            #                     )(input_layer)
            _u = self.normalizer()(_u)
            if self.dropout_rate_generator_up_sampling > 0:
                _u = Dropout(rate=self.dropout_rate_generator_up_sampling, seed=1234)(_u)
            _u = Concatenate()([_u, skip_layer])
        else:
            _u: keras_tensor.KerasTensor = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_layer)
            _u = Conv2D(filters=n_filters,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.initializer
                        )(_u)
            #_u = Conv2DTranspose(filters=n_filters,
            #                     kernel_size=(3, 3),
            #                     strides=(2, 2),
            #                     padding='same',
            #                     kernel_initializer=self.initializer
            #                     )(input_layer)
            _u = self.normalizer()(_u)
            if self.dropout_rate_generator_up_sampling > 0:
                _u = Dropout(rate=self.dropout_rate_generator_up_sampling, seed=1234)(_u)
        _u = ReLU(max_value=None, negative_slope=0, threshold=0)(_u)
        return _u

    def _convolutional_layer_generator_encoder(self,
                                               input_layer: keras_tensor.KerasTensor,
                                               n_filters: int
                                               ) -> keras_tensor.KerasTensor:
        """
        Convolutional layer encoder for down-sampling

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        if self.generator_type == 'u':
            _kernel_size: tuple = (4, 4)
        else:
            _kernel_size: tuple = (3, 3)
        _d: keras_tensor.KerasTensor = Conv2D(filters=n_filters,
                                              kernel_size=_kernel_size,
                                              strides=(2, 2),
                                              padding='same',
                                              kernel_initializer=self.initializer
                                              )(input_layer)
        _d = self.normalizer()(_d)
        if self.dropout_rate_generator_down_sampling > 0:
            _d = Dropout(rate=self.dropout_rate_generator_down_sampling, seed=1234)(_d)
        _d = LeakyReLU(alpha=0.2)(_d)
        return _d

    def _eval_training(self, file_path: str):
        """
        Evaluate current training by generating predictions based on test images

        :param file_path: str
            Complete file path to save test predictions
        """
        _image_processor: ImageProcessor = ImageProcessor(file_path_clean_images='',
                                                          file_path_noisy_images=self.file_path_eval_noisy_data,
                                                          file_path_multi_noisy_images=self.file_path_moe_noisy_images,
                                                          n_channels=self.n_channels,
                                                          batch_size=self.batch_size,
                                                          image_resolution=(self.image_width, self.image_height),
                                                          normalize=self.normalize,
                                                          flip=False,
                                                          crop=self.kwargs.get('crop')
                                                          )
        for image_noisy_file_path, image_noisy in _image_processor.load_images(n_images=1, label=self.clf_label):
            _fake_noisy: np.array = self.generator_A.predict(image_noisy)
            _output_file_path_fake: str = os.path.join(file_path,
                                                       f"test_fake_{image_noisy_file_path[0].split('/')[-1]}"
                                                       )
            _output_file_path_noisy: str = os.path.join(file_path,
                                                        f"test_B_{image_noisy_file_path[0].split('/')[-1]}"
                                                        )
            self.image_processor.save_image(image=np.array(_fake_noisy).squeeze(), output_file_path=_output_file_path_fake)
            self.image_processor.save_image(image=np.array(image_noisy).squeeze(), output_file_path=_output_file_path_noisy)
            Log().log(msg=f'Save evaluation image: {_output_file_path_fake}')

    def _generate_training_report(self) -> None:
        """
        Generate cycle-gan training report
        """
        _cycle_gan_architecture: str = 'Discriminator: PatchGAN | '
        if self.generator_type == 'u':
            _cycle_gan_architecture = f'{_cycle_gan_architecture}Generator: U-Network'
        else:
            _cycle_gan_architecture = f'{_cycle_gan_architecture}Generator: Residual Network {self.n_res_net_blocks} Blocks'
        self.training_report = dict(cycle_gan_architecture=_cycle_gan_architecture,
                                    training_type=self.training_type,
                                    label=self.label,
                                    training_time=self.training_time,
                                    elapsed_time=self.elapsed_time,
                                    learning_rate=self.learning_rate,
                                    epoch=self.epoch,
                                    batch=self.batch,
                                    discriminator_batch_size=self.discriminator_batch_size,
                                    generator_batch_size=self.generator_batch_size,
                                    batch_size=self.batch_size,
                                    batches=self.image_processor.n_batches,
                                    image_samples=self.batch_size * self.image_processor.n_batches,
                                    discriminator_loss=self.discriminator_loss,
                                    discriminator_accuracy=self.discriminator_accuracy,
                                    generator_loss=self.generator_loss,
                                    adversarial_loss=self.adversarial_loss,
                                    reconstruction_loss=self.reconstruction_loss,
                                    identity_loss=self.identy_loss
                                    )

    def _residual_network(self):
        """
        Residual network generator using encoder - transformer - decoder convolutional network architecture
        """
        _input: Input = Input(shape=self.image_shape)
        _n_filters: int = self.start_n_filters_generator
        _g: keras_tensor.KerasTensor = ReflectionPadding2D(padding=(3, 3))(_input)
        _g = Conv2D(filters=self.start_n_filters_generator,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer=self.initializer
                    )(_g)
        _g = self.normalizer()(_g)
        _g = LeakyReLU(alpha=0.2)(_g)
        # Encoder (Down-Sampling / Feature Extraction):
        for i in range(0, self.n_conv_layers_generator_res_net, 1):
            if _n_filters < self.max_n_filters_generator:
                if self.up_sample_n_filters_period > 0:
                    if i % self.up_sample_n_filters_period == 0:
                        _n_filters *= 2
                else:
                    _n_filters *= 2
            _g = self._convolutional_layer_generator_encoder(input_layer=_g, n_filters=_n_filters)
        # Transformer (Residual Network Blocks):
        for _ in range(0, self.n_res_net_blocks, 1):
            _g = self._residual_network_block(input_layer=_g, n_filters=_n_filters)
        # Decoder (Up-Sampling):
        for j in range(0, self.n_conv_layers_generator_res_net, 1):
            if _n_filters > self.start_n_filters_generator:
                if self.up_sample_n_filters_period > 0:
                    if j % self.up_sample_n_filters_period == 0:
                        _n_filters //= 2
                else:
                    _n_filters //= 2
            _g = self._convolutional_layer_generator_decoder(input_layer=_g, skip_layer=None, n_filters=_n_filters)
        _g = ReflectionPadding2D(padding=(3, 3))(_g)
        _fake_image: keras_tensor.KerasTensor = Conv2D(filters=self.n_channels,
                                                       kernel_size=(7, 7),
                                                       strides=(1, 1),
                                                       padding='valid',
                                                       kernel_initializer=self.initializer,
                                                       activation='tanh'
                                                       )(_g)
        return Model(inputs=_input, outputs=_fake_image, name=self.model_name)

    def _residual_network_block(self,
                                input_layer: keras_tensor.KerasTensor,
                                n_filters: int
                                ) -> keras_tensor.KerasTensor:
        """
        Residual network block

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _r: keras_tensor.KerasTensor = ReflectionPadding2D(padding=(1, 1))(input_layer)
        _r = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer=self.initializer
                    )(_r)
        _r = self.normalizer()(_r)
        if self.dropout_rate_generator_res_net > 0:
            _r = Dropout(rate=self.dropout_rate_generator_res_net)(_r)
        _r = ReLU(max_value=None, negative_slope=0, threshold=0)(_r)
        _r = ReflectionPadding2D(padding=(1, 1))(_r)
        _r = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer=self.initializer
                    )(_r)
        _r = self.normalizer()(_r)
        if self.dropout_rate_generator_res_net > 0:
            _r = Dropout(rate=self.dropout_rate_generator_res_net)(_r)
        _r = Add()([_r, input_layer])
        _r = ReLU(max_value=None, negative_slope=0, threshold=0)(_r)
        return _r

    def _save_models(self, model_output_path: str):
        """
        Save cycle-gan models

        :param model_output_path: str
            Complete file path of the model output
        """
        self.generator_A.save(filepath=f'{self.model_name_generator_A}_h5')
        save_file_to_s3(file_path=model_output_path, obj=None, input_file_path=f'{self.model_name_generator_A}_h5')
        self.generator_B.save(filepath=f'{self.model_name_generator_B}_h5')
        save_file_to_s3(file_path=model_output_path, obj=None, input_file_path=f'{self.model_name_generator_B}_h5')
        self.discriminator_A.save(filepath=f'{self.model_name_discriminator_A}_h5')
        save_file_to_s3(file_path=model_output_path, obj=None, input_file_path=f'{self.model_name_discriminator_A}_h5')
        self.discriminator_B.save(filepath=f'{self.model_name_discriminator_B}_h5')
        save_file_to_s3(file_path=model_output_path, obj=None, input_file_path=f'{self.model_name_discriminator_B}_h5')
        self.combined_model.save(filepath=f'{self.model_name_combined_model}_h5')
        save_file_to_s3(file_path=model_output_path, obj=None, input_file_path=f'{self.model_name_combined_model}_h5')

    def _u_network(self) -> Model:
        """
        U-network

        :return: Model
            U-network model
        """
        _input: Input = Input(shape=self.image_shape)
        _n_filters: int = self.start_n_filters_generator
        _g: keras_tensor.KerasTensor = Conv2D(filters=_n_filters,
                                              kernel_size=(3, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              kernel_initializer=self.initializer
                                              )(_input)
        _g = self.normalizer()(_g)
        _g = LeakyReLU(alpha=0.2)(_g)
        _u_net_layers: list = [_g]
        # Encoder (Down-Sampling / Feature Extraction):
        for i in range(0, self.n_conv_layers_generator_u_net, 1):
            if _n_filters < self.max_n_filters_generator:
                if self.up_sample_n_filters_period > 0:
                    if i % self.up_sample_n_filters_period == 0:
                        _n_filters *= 2
                else:
                    _n_filters *= 2
            _u_net_layers.append(self._convolutional_layer_generator_encoder(input_layer=_u_net_layers[-1],
                                                                             n_filters=_n_filters
                                                                             )
                                 )
        # Decoder (Up-Sampling):
        for j in range(0, self.n_conv_layers_generator_u_net, 1):
            if _n_filters > self.start_n_filters_generator:
                if self.up_sample_n_filters_period > 0:
                    if j % self.up_sample_n_filters_period == 0:
                        _n_filters //= 2
                else:
                    _n_filters //= 2
            _j: int = -2 - (j * 2)
            _u_net_layers.append(self._convolutional_layer_generator_decoder(input_layer=_u_net_layers[-1],
                                                                             skip_layer=_u_net_layers[_j],
                                                                             n_filters=_n_filters
                                                                             )
                                 )
        _g = _u_net_layers[-1]
        _fake_image: keras_tensor.KerasTensor = Conv2D(filters=self.n_channels,
                                                       kernel_size=(7, 7),
                                                       strides=(1, 1),
                                                       padding='same',
                                                       kernel_initializer=self.initializer,
                                                       activation='tanh'
                                                       )(_g)
        return Model(inputs=_input, outputs=_fake_image, name=self.model_name)

    def inference(self,
                  file_path_generator: str,
                  file_path_noisy_images: str,
                  file_path_cleaned_images: str,
                  file_suffix: str = 'cleaned'
                  ):
        """
        Clean noisy document images based on training

        :param file_path_generator: str
            Complete file path of trained generator A

        :param file_path_noisy_images: str
            Complete file path of noisy images to clean

        :param file_path_cleaned_images: str
            Complete file path of the output (cleaned / denoised images)

        :param file_suffix: str
            Suffix of the output file name
        """
        _image_processor: ImageProcessor = ImageProcessor(file_path_clean_images='',
                                                          file_path_noisy_images=file_path_noisy_images,
                                                          n_channels=self.n_channels,
                                                          batch_size=self.batch_size,
                                                          image_resolution=(256, 256),
                                                          flip=False,
                                                          crop=None
                                                          )
        if len(file_path_generator) > 0:
            _generator_A_json_file: str = file_path_generator.replace('h5', 'json')
            json_file = open(_generator_A_json_file, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.generator_A: Model = keras.models.model_from_json(loaded_model_json,
                                                                   custom_objects={"InstanceNormalization": InstanceNormalization})
            # load weights into new model
            self.generator_A.load_weights(file_path_generator)
            #self.generator_A: Model = load_model(filepath=file_path_generator,
            #                                     custom_objects={"InstanceNormalization": InstanceNormalization},
            #                                     compile=True,
            #                                     options=None
            #                                     )
        for image_file_name, image in _image_processor.load_images():
            if file_suffix is None or len(file_suffix) == 0:
                _output_file_path: str = os.path.join(file_path_cleaned_images, image_file_name[0].split('/')[-1])
            else:
                _output_file_path: str = os.path.join(file_path_cleaned_images,
                                                      f"{file_suffix}_{image_file_name[0].split('/')[-1]}"
                                                      )
            _prediction: np.array = self.generator_A.predict(x=image[0])
            _image_processor.save_image(image=np.array(_prediction).squeeze(),
                                        output_file_path=_output_file_path
                                        )

    def train(self,
              model_output_path: str,
              eval_output_path: str,
              n_epoch: int = 300,
              asynchron: bool = False,
              discriminator_batch_size: int = 100,
              generator_batch_size: int = 10,
              early_stopping_batch: int = 0,
              random_noise_types: bool = False,
              checkpoint_epoch_interval: int = 5,
              evaluation_epoch_interval: int = 1
              ):
        """
        Train cycle-gan models

        :param model_output_path: str
            Complete file path of the model output

        :param eval_output_path: str
            Path of the evaluation output images

        :param n_epoch: int
            Number of epochs to train

        :param asynchron: bool
            Whether to train discriminator and generator asynchron or synchron

        :param discriminator_batch_size: int
            Batch size of to update discriminator

        :param generator_batch_size: int
            Batch size of to update generator

        :param early_stopping_batch: int
            Number of batch to process before to stop epoch early

        :param random_noise_types: bool
            Whether to load images containing different noise types randomly or systematically (moe only)

        :param checkpoint_epoch_interval: int
            Number of epoch intervals for saving model checkpoint

        :param evaluation_epoch_interval: int
            Number of epoch intervals for evaluating model training
        """
        # Build Cycle-GAN Network:
        self._build_cycle_gan_network()
        _t0: datetime = datetime.datetime.now()
        _discriminator_batch_real_A: List[np.array] = []
        _discriminator_batch_fake_A: List[np.array] = []
        _discriminator_batch_real_B: List[np.array] = []
        _discriminator_batch_fake_B: List[np.array] = []
        _generator_batch_real: List[np.array] = []
        _generator_batch_noisy: List[np.array] = []
        _n_updates_discriminator: int = 0
        _n_updates_generator: int = 0
        _print_losses: bool = False
        # Adversarial loss ground truths:
        if asynchron:
            self.training_type = 'asynchron'
            self.discriminator_batch_size = discriminator_batch_size if discriminator_batch_size > 0 else 50
            self.generator_batch_size = generator_batch_size if generator_batch_size > 0 else 50
            _valid: np.array = np.ones((1, ) + self.discriminator_patch)
            _fake: np.array = np.zeros((1, ) + self.discriminator_patch)
        else:
            self.training_type = 'synchron'
            _valid: np.array = np.ones((self.batch_size, ) + self.discriminator_patch)
            _fake: np.array = np.zeros((self.batch_size, ) + self.discriminator_patch)
        _label: int = -1
        for epoch in range(n_epoch):
            if not random_noise_types:
                if _label + 1 == len(self.file_path_moe_noisy_images):
                    _label = 0
                else:
                    _label += 1
            for batch_i, (images_A, images_B, label) in enumerate(self.image_processor.load_batch(label=_label)):
                self.clf_label = label
                self.label.append(self.clf_label)
                # Translate images to opposite domain:
                _fake_B = self.generator_A.predict(x=images_B)
                _fake_A = self.generator_B.predict(x=images_A)
                if asynchron:
                    _discriminator_batch_real_A.append(images_A)
                    _discriminator_batch_real_B.append(images_B)
                    _generator_batch_real.append(images_A)
                    _generator_batch_noisy.append(images_B)
                    _discriminator_batch_fake_A.append(_fake_A)
                    _discriminator_batch_fake_B.append(_fake_B)
                    if len(_discriminator_batch_real_A) == self.discriminator_batch_size:
                        _n_updates_discriminator += 1
                        for (d_real_A, d_fake_A, d_real_B, d_fake_B) in zip(_discriminator_batch_real_A,
                                                                            _discriminator_batch_fake_A,
                                                                            _discriminator_batch_real_B,
                                                                            _discriminator_batch_fake_B
                                                                            ):
                            # Train the discriminators (original images = real / translated = Fake):
                            _discriminator_loss_real_A = self.discriminator_A.train_on_batch(d_real_A, _valid)
                            _discriminator_loss_fake_A = self.discriminator_A.train_on_batch(d_fake_A, _fake)
                            _discriminator_loss_A = 0.5 * np.add(_discriminator_loss_real_A, _discriminator_loss_fake_A)
                            _discriminator_loss_real_B = self.discriminator_B.train_on_batch(d_real_B, _valid)
                            _discriminator_loss_fake_B = self.discriminator_B.train_on_batch(d_fake_B, _fake)
                            _discriminator_loss_B = 0.5 * np.add(_discriminator_loss_real_B, _discriminator_loss_fake_B)
                            # Total discriminator loss:
                            _discriminator_loss = 0.5 * np.add(_discriminator_loss_A, _discriminator_loss_B)
                            self.discriminator_loss.append(round(_discriminator_loss[0], 8))
                            self.discriminator_accuracy.append(round(100 * _discriminator_loss[1], 4))
                        _discriminator_batch_real_A = []
                        _discriminator_batch_real_B = []
                        _discriminator_batch_fake_A = []
                        _discriminator_batch_fake_B = []
                        _print_losses = True
                    if len(_generator_batch_real) == self.generator_batch_size:
                        _n_updates_generator += 1
                        for (g_real, g_noisy) in zip(_generator_batch_real, _generator_batch_noisy):
                            # Train the generators:
                            _generator_loss = self.combined_model.train_on_batch([g_real,
                                                                                  g_noisy
                                                                                  ],
                                                                                 [_valid,
                                                                                  _valid,
                                                                                  g_real,
                                                                                  g_noisy,
                                                                                  g_real,
                                                                                  g_noisy
                                                                                  ]
                                                                                 )
                            self.generator_loss.append(round(_generator_loss[0], 8))
                            self.adversarial_loss.append(round(np.mean(_generator_loss[1:3]), 8))
                            self.reconstruction_loss.append(round(np.mean(_generator_loss[3:5]), 8))
                            self.identy_loss.append(round(np.mean(_generator_loss[5:6]), 8))
                        _generator_batch_real = []
                        _generator_batch_noisy = []
                        _print_losses = True
                else:
                    # Train the discriminators (original images = real / translated = Fake):
                    _discriminator_loss_real_A = self.discriminator_A.train_on_batch(images_A, _valid)
                    _discriminator_loss_fake_A = self.discriminator_A.train_on_batch(_fake_A, _fake)
                    _discriminator_loss_A = 0.5 * np.add(_discriminator_loss_real_A, _discriminator_loss_fake_A)
                    _discriminator_loss_real_B = self.discriminator_B.train_on_batch(images_B, _valid)
                    _discriminator_loss_fake_B = self.discriminator_B.train_on_batch(_fake_B, _fake)
                    _discriminator_loss_B = 0.5 * np.add(_discriminator_loss_real_B, _discriminator_loss_fake_B)
                    # Total discriminator loss:
                    _discriminator_loss = 0.5 * np.add(_discriminator_loss_A, _discriminator_loss_B)
                    # Train the generators:
                    _generator_loss = self.combined_model.train_on_batch([images_A,
                                                                          images_B
                                                                          ],
                                                                         [_valid,
                                                                          _valid,
                                                                          images_A,
                                                                          images_B,
                                                                          images_A,
                                                                          images_B
                                                                          ]
                                                                         )
                _elapsed_time: datetime = datetime.datetime.now() - _t0
                self.elapsed_time.append(str(_elapsed_time))
                self.epoch.append(epoch)
                self.batch.append(batch_i)
                self.discriminator_loss.append(round(_discriminator_loss[0], 8))
                self.discriminator_accuracy.append(round(100 * _discriminator_loss[1], 4))
                self.generator_loss.append(round(_generator_loss[0], 8))
                self.adversarial_loss.append(round(np.mean(_generator_loss[1:3]), 8))
                self.reconstruction_loss.append(round(np.mean(_generator_loss[3:5]), 8))
                self.identy_loss.append(round(np.mean(_generator_loss[5:6]), 8))
                # Print training progress:
                if asynchron:
                    if _print_losses and len(self.discriminator_loss) > 0 and len(self.generator_loss) > 0:
                        _print_epoch_status: str = f'[Epoch: {epoch}/{n_epoch}]'
                        _print_batch_status: str = f'[Batch: {batch_i}/{self.image_processor.n_batches}]'
                        _print_discriminator_loss_status: str = f'[D loss: {self.discriminator_loss[-1]}, acc: {self.discriminator_accuracy[-1]}]'
                        _print_generator_loss_status: str = f'[G loss: {self.generator_loss[-1]}, adv: {self.adversarial_loss[-1]}, recon: {self.reconstruction_loss[-1]}, id: {self.identy_loss[-1]}]'
                        Log().log(f'{_print_epoch_status}{_print_batch_status}{_print_discriminator_loss_status}{_print_generator_loss_status} time: {_elapsed_time}')
                        _print_losses = False
                else:
                    _print_epoch_status: str = f'[Epoch: {epoch}/{n_epoch}]'
                    _print_batch_status: str = f'[Batch: {batch_i}/{self.image_processor.n_batches}]'
                    _print_discriminator_loss_status: str = f'[D loss: {self.discriminator_loss[-1]}, acc: {self.discriminator_accuracy[-1]}]'
                    _print_generator_loss_status: str = f'[G loss: {self.generator_loss[-1]}, adv: {self.adversarial_loss[-1]}, recon: {self.reconstruction_loss[-1]}, id: {self.identy_loss[-1]}]'
                    Log().log(f'{_print_epoch_status}{_print_batch_status}{_print_discriminator_loss_status}{_print_generator_loss_status} time: {_elapsed_time}')
                if early_stopping_batch > 0:
                    if batch_i + 1 == early_stopping_batch:
                        break
            # Save checkpoint:
            if checkpoint_epoch_interval > 0:
                if (epoch % checkpoint_epoch_interval == 0) and (epoch > 0):
                    self._save_models(model_output_path=model_output_path)
            # Evaluate current training:
            if self.file_path_eval_noisy_data is not None and len(self.file_path_eval_noisy_data) > 0:
                if epoch % evaluation_epoch_interval == 0:
                    self._eval_training(file_path=eval_output_path)
        # Save fully trained models:
        self._save_models(model_output_path=model_output_path)
        # Generate training report:
        self.training_time = self.elapsed_time[-1]
        self._generate_training_report()
