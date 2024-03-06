"""

Task: ... (Function to run in container)

"""

import argparse
import ast

from auto_encoder import AutoEncoder
from custom_logger import Log
from cycle_gan import CycleGAN
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import List


PARSER = argparse.ArgumentParser(description="image translation")
PARSER.add_argument('-unpaired', type=int, required=False, default=1, help='whether to have paired or unpaired data')
PARSER.add_argument('-train_data_set_a_path', type=str, required=True, default=None, help='file path of the training data set for image type A')
PARSER.add_argument('-train_data_set_b_path', type=str, required=True, default=None, help='file path of the training data set for image type B')
PARSER.add_argument('-test_data_set_b_path', type=str, required=False, default=None, help='file path of the test data set for image type B')
PARSER.add_argument('-n_channels', type=int, required=False, default=3, help='number of image channels to use')
PARSER.add_argument('-image_height', type=int, required=False, default=256, help='height of the images')
PARSER.add_argument('-image_width', type=int, required=False, default=256, help='width of the images')
PARSER.add_argument('-learning_rate', type=float, required=False, default=0.0002, help='learning rate of the neural network optimizer component')
PARSER.add_argument('-optimizer', type=str, required=False, default='adam', help='name of the optimizer')
PARSER.add_argument('-initializer', type=str, required=False, default='he_normal', help='name of the initializer')
PARSER.add_argument('-batch_size', type=int, required=False, default=1, help='batch size')
PARSER.add_argument('-n_epoch', type=int, required=False, default=300, help='number of epochs to train')
PARSER.add_argument('-start_n_filters_auto_encoder', type=int, required=False, default=64, help='number of filters used in first convolutional layer in auto-encoder network')
PARSER.add_argument('-n_conv_layers_auto_encoder', type=int, required=False, default=3, help='number of convolutional layers in auto-encoder network')
PARSER.add_argument('-dropout_rate_auto_encoder', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer in auto-encoder network')
PARSER.add_argument('-start_n_filters_discriminator', type=int, required=False, default=64, help='number of filters used in first convolutional layer in discriminator network')
PARSER.add_argument('-max_n_filters_discriminator', type=int, required=False, default=512, help='maximum number of filter used in all convolutional layers in discriminator network')
PARSER.add_argument('-n_conv_layers_discriminator', type=int, required=False, default=3, help='number of convolutional layers in discriminator network')
PARSER.add_argument('-dropout_rate_discriminator', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer in discriminator network')
PARSER.add_argument('-start_n_filters_generator', type=int, required=False, default=32, help='number of filters used in first convolutional layer in generator network')
PARSER.add_argument('-max_n_filters_generator', type=int, required=False, default=512, help='maximum number of filter used in all convolutional layers in generator network')
PARSER.add_argument('-up_sample_n_filters_period', type=int, required=False, default=0, help='number of layers until up-sampling number of filters')
PARSER.add_argument('-generator_type', type=str, required=False, default='res', help='abbreviated name of the generator type')
PARSER.add_argument('-n_res_net_blocks', type=int, required=False, default=6, help='number of residual network blocks')
PARSER.add_argument('-n_conv_layers_generator_res_net', type=int, required=False, default=2, help='number of convolutional layers in residual network generator')
PARSER.add_argument('-n_conv_layers_generator_u_net', type=int, required=False, default=3, help='number of convolutional layers in u-network generator')
PARSER.add_argument('-dropout_rate_generator_res_net', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer in residual network generator')
PARSER.add_argument('-dropout_rate_generator_down_sampling', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer in generator down-sampling network')
PARSER.add_argument('-dropout_rate_generator_up_sampling', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer in generator up-sampling network')
PARSER.add_argument('-asynchron', type=int, required=False, default=0, help='whether to train cycle gan network asynchronously or not')
PARSER.add_argument('-discriminator_batch_size', type=int, required=False, default=100, help='batch size of the discriminator network update in asynchron mode')
PARSER.add_argument('-generator_batch_size', type=int, required=False, default=10, help='batch size of the generator network update in asynchron mode')
PARSER.add_argument('-early_stopping_batch', type=int, required=False, default=0, help='early stopping batch number')
PARSER.add_argument('-checkpoint_epoch_interval', type=int, required=False, default=5, help='epoch interval of saving model checkpoint')
PARSER.add_argument('-evaluation_epoch_interval', type=int, required=False, default=1, help='epoch interval of evaluating model')
PARSER.add_argument('-print_model_architecture', type=int, required=False, default=1, help='whether to print model architecture or not')
PARSER.add_argument('-kwargs', type=str, required=False, default=None, help='key-word arguments')
PARSER.add_argument('-s3_output_file_path_evaluation', type=str, required=True, default=None, help='S3 file path of the evaluation images output')
PARSER.add_argument('-s3_output_file_path_model_artifact', type=str, required=False, default=None, help='S3 file path of the model artifact output')
ARGS = PARSER.parse_args()


def image_translation(train_data_set_a_path: str,
                      train_data_set_b_path: str,
                      test_data_set_b_path: str,
                      s3_output_file_path_evaluation: str,
                      unpaired: bool = True,
                      n_channels: int = 3,
                      image_height: int = 256,
                      image_width: int = 256,
                      learning_rate: float = 0.0002,
                      optimizer: str = 'adam',
                      initializer: str = 'he_normal',
                      batch_size: int = 1,
                      n_epoch: int = 300,
                      start_n_filters_auto_encoder: int = 64,
                      n_conv_layers_auto_encoder: int = 3,
                      dropout_rate_auto_encoder: float = 0.0,
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
                      asynchron: bool = False,
                      discriminator_batch_size: int = 100,
                      generator_batch_size: int = 10,
                      early_stopping_batch: int = 0,
                      checkpoint_epoch_interval: int = 5,
                      evaluation_epoch_interval: int = 1,
                      print_model_architecture: bool = True,
                      s3_output_file_path_model_artifact: str = None,
                      **kwargs
                      ) -> None:
    """
    Translate paired or unpaired image characteristics

    :param train_data_set_a_path: str
        File path of the training data set for images type A

    :param train_data_set_b_path: str
        File path of the training data set for images type B

    :param test_data_set_b_path: str
        File path of the test data set for images type B

    :param s3_output_file_path_evaluation: str
        Path of the evaluation output images

    :param unpaired: bool
        Whether to have paired or unpaired data set for image translation
            -> paired: Auto-Encoder
            -> unpaired: Cycle GAN

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

    :param optimizer: str
        Name of the optimizer
            -> adam: Adam
            -> rmsprop: RmsProp
            -> sgd: Stochastic Gradient Descent

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

    :param n_epoch: int
        Number of epochs

    :param start_n_filters_auto_encoder: int
        Number of filters used in first convolutional layer in auto-encoder network

    :param n_conv_layers_auto_encoder: int
        Number of convolutional layers in auto-encoder network

    :param dropout_rate_auto_encoder: float
        Dropout rate used after each convolutional layer in auto-encoder network

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
        Abbreviated name of the generator type
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

    :param asynchron: bool
            Whether to train discriminator and generator asynchron or synchron

    :param discriminator_batch_size: int
        Batch size of to update discriminator

    :param generator_batch_size: int
        Batch size of to update generator

    :param early_stopping_batch: int
        Number of batch to process before to stop epoch early

    :param checkpoint_epoch_interval: int
        Number of epoch intervals for saving model checkpoint

    :param evaluation_epoch_interval: int
        Number of epoch intervals for evaluating model training

    :param print_model_architecture: bool
        Whether to print architecture of cycle-gan model components (discriminators & generators) or not

    :param s3_output_file_path_model_artifact: str
        Complete file path of the model artifact

    :param kwargs: dict
        Key-word arguments for class ImageProcessor and compiling model configuration
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    if unpaired:
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=train_data_set_a_path,
                                        file_path_train_noisy_images=train_data_set_b_path,
                                        file_path_eval_noisy_images=test_data_set_b_path,
                                        n_channels=n_channels,
                                        image_height=image_height,
                                        image_width=image_width,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer,
                                        initializer=initializer,
                                        batch_size=batch_size,
                                        start_n_filters_discriminator=start_n_filters_discriminator,
                                        max_n_filters_discriminator=max_n_filters_discriminator,
                                        n_conv_layers_discriminator=n_conv_layers_discriminator,
                                        dropout_rate_discriminator=dropout_rate_discriminator,
                                        start_n_filters_generator=start_n_filters_generator,
                                        max_n_filters_generator=max_n_filters_generator,
                                        up_sample_n_filters_period=up_sample_n_filters_period,
                                        generator_type=generator_type,
                                        n_res_net_blocks=n_res_net_blocks,
                                        n_conv_layers_generator_res_net=n_conv_layers_generator_res_net,
                                        n_conv_layers_generator_u_net=n_conv_layers_generator_u_net,
                                        dropout_rate_generator_res_net=dropout_rate_generator_res_net,
                                        dropout_rate_generator_down_sampling=dropout_rate_generator_down_sampling,
                                        dropout_rate_generator_up_sampling=dropout_rate_generator_up_sampling,
                                        print_model_architecture=print_model_architecture,
                                        **kwargs
                                        )
        _cycle_gan.train(model_output_path=s3_output_file_path_model_artifact,
                         eval_output_path=s3_output_file_path_evaluation,
                         n_epoch=n_epoch,
                         asynchron=asynchron,
                         discriminator_batch_size=discriminator_batch_size,
                         generator_batch_size=generator_batch_size,
                         early_stopping_batch=early_stopping_batch,
                         checkpoint_epoch_interval=checkpoint_epoch_interval,
                         evaluation_epoch_interval=evaluation_epoch_interval
                         )
    else:
        _auto_encoder: AutoEncoder = AutoEncoder(file_path_train_clean_images=train_data_set_a_path,
                                                 file_path_train_noisy_images=train_data_set_b_path,
                                                 n_channels=n_channels,
                                                 image_height=image_height,
                                                 image_width=image_width,
                                                 learning_rate=learning_rate,
                                                 optimizer=optimizer,
                                                 initializer=initializer,
                                                 batch_size=batch_size,
                                                 start_n_filters=start_n_filters_auto_encoder,
                                                 n_conv_layers=n_conv_layers_auto_encoder,
                                                 dropout_rate=dropout_rate_auto_encoder,
                                                 print_model_architecture=print_model_architecture
                                                 )
        _auto_encoder.train(model_output_path=s3_output_file_path_model_artifact,
                            n_epoch=n_epoch,
                            early_stopping_patience=early_stopping_batch
                            )
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)


if __name__ == '__main__':
    if ARGS.kwargs:
        ARGS.kwargs = ast.literal_eval(ARGS.kwargs)
    image_translation(train_data_set_a_path=ARGS.train_data_set_a_path,
                      train_data_set_b_path=ARGS.train_data_set_b_path,
                      test_data_set_b_path=ARGS.test_data_set_b_path,
                      s3_output_file_path_evaluation=ARGS.s3_output_file_path_evaluation,
                      unpaired=bool(ARGS.unpaired),
                      n_channels=ARGS.n_channels,
                      image_height=ARGS.image_height,
                      image_width=ARGS.image_width,
                      learning_rate=ARGS.learning_rate,
                      optimizer=ARGS.optimizer,
                      initializer=ARGS.initializer,
                      batch_size=ARGS.batch_size,
                      n_epoch=ARGS.n_epoch,
                      start_n_filters_auto_encoder=ARGS.start_n_filters_auto_encoder,
                      n_conv_layers_auto_encoder=ARGS.n_conv_layers_auto_encoder,
                      dropout_rate_auto_encoder=ARGS.dropout_rate_auto_encoder,
                      start_n_filters_discriminator=ARGS.start_n_filters_discriminator,
                      max_n_filters_discriminator=ARGS.max_n_filters_discriminator,
                      n_conv_layers_discriminator=ARGS.n_conv_layers_discriminator,
                      dropout_rate_discriminator=ARGS.dropout_rate_discriminator,
                      start_n_filters_generator=ARGS.start_n_filters_generator,
                      max_n_filters_generator=ARGS.max_n_filters_generator,
                      up_sample_n_filters_period=ARGS.up_sample_n_filters_period,
                      generator_type=ARGS.generator_type,
                      n_res_net_blocks=ARGS.n_res_net_blocks,
                      n_conv_layers_generator_res_net=ARGS.n_conv_layers_generator_res_net,
                      n_conv_layers_generator_u_net=ARGS.n_conv_layers_generator_u_net,
                      dropout_rate_generator_res_net=ARGS.dropout_rate_generator_res_net,
                      dropout_rate_generator_down_sampling=ARGS.dropout_rate_generator_down_sampling,
                      dropout_rate_generator_up_sampling=ARGS.dropout_rate_generator_up_sampling,
                      asynchron=bool(ARGS.asynchron),
                      discriminator_batch_size=ARGS.discriminator_batch_size,
                      generator_batch_size=ARGS.generator_batch_size,
                      early_stopping_batch=ARGS.early_stopping_batch,
                      checkpoint_epoch_interval=ARGS.checkpoint_epoch_interval,
                      evaluation_epoch_interval=ARGS.evaluation_epoch_interval,
                      print_model_architecture=bool(ARGS.print_model_architecture),
                      s3_output_file_path_model_artifact=ARGS.s3_output_file_path_model_artifact,
                      **ARGS.kwargs
                      )
