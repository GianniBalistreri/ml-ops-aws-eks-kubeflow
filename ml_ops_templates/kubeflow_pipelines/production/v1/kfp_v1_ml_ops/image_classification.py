"""

Kubeflow pipeline component: Image classification

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List


def image_classification(train_data_set_path: str,
                         test_data_set_path: str,
                         labels: List[str],
                         s3_output_path_evaluation_train_data: str,
                         s3_output_path_evaluation_test_data: str,
                         s3_output_file_path_model_artifact: str,
                         aws_account_id: str,
                         aws_region: str,
                         n_channels: int = 3,
                         image_height: int = 256,
                         image_width: int = 256,
                         learning_rate: float = 0.001,
                         optimizer: str = 'adam',
                         initializer: str = 'he_normal',
                         activation: str = 'relu',
                         batch_size: int = 32,
                         n_epoch: int = 10,
                         n_conv_layers: int = 7,
                         start_n_filters_conv_layers: int = 32,
                         max_n_filters_conv_layers: int = 512,
                         dropout_rate_conv_layers: float = 0.0,
                         up_size_n_filters_period: int = 3,
                         pool_size_conv_layers: int = 2,
                         val_data_set_path: str = None,
                         s3_output_path_evaluation_val_data: str = None,
                         checkpoint_epoch_interval: int = 5,
                         evaluation_epoch_interval: int = 1,
                         print_model_architecture: bool = True,
                         docker_image_name: str = 'ml-ops-image-classification',
                         docker_image_tag: str = 'v1',
                         volume: dsl.VolumeOp = None,
                         volume_dir: str = '/mnt',
                         display_name: str = 'Image Classification',
                         n_cpu_request: str = None,
                         n_cpu_limit: str = None,
                         n_gpu: str = None,
                         gpu_vendor: str = 'nvidia',
                         memory_request: str = '1G',
                         memory_limit: str = None,
                         ephemeral_storage_request: str = '5G',
                         ephemeral_storage_limit: str = None,
                         instance_name: str = 'm5.xlarge',
                         max_cache_staleness: str = 'P0D',
                         **kwargs
                         ) -> dsl.ContainerOp:
    """
    Classify images

    :param train_data_set_path: str
        File path of the training images

    :param test_data_set_path: str
        File path of the test images

    :param labels: List[str]
        Class labels

    :param s3_output_path_evaluation_train_data: str
        Complete file path of the evaluation train data

    :param s3_output_path_evaluation_test_data: str
        Complete file path of the evaluation test data

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

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

    :param activation: str
        Name of the activation function used in convolutional neural network layers

    :param batch_size: int
        Batch size

    :param n_epoch: int
        Number of epochs

    :param n_conv_layers: int
        Number of convolutional neural network layers

    :param start_n_filters_conv_layers: int
        Number of convolutional layers in first convolutional layer

    :param max_n_filters_conv_layers: int
        Maximum number of filter used in all convolutional layers in discriminator network

    :param dropout_rate_conv_layers: float
        Dropout rate used in convolutional layer

    :param up_size_n_filters_period: int
        Period to up-size number of filters used in convolutional layers

    :param pool_size_conv_layers: int
        Number of strides used in max pooling layer

    :param val_data_set_path: str
        File path of the validation images

    :param s3_output_path_evaluation_val_data: str
        Complete file path of the evaluation validation data

    :param checkpoint_epoch_interval: int
        Number of epoch intervals for saving model checkpoint

    :param evaluation_epoch_interval: int
        Number of epoch intervals for evaluating model training

    :param print_model_architecture: bool
        Whether to print architecture of classification model components or not

    :param s3_output_file_path_model_artifact: str
        Complete file path of the model artifact

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :param docker_image_name: str
        Name of the docker image repository

    :param docker_image_tag: str
        Name of the docker image tag

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :param display_name: str
        Display name of the Kubeflow Pipeline component

    :param n_cpu_request: str
        Number of requested CPU's

    :param n_cpu_limit: str
        Maximum number of requested CPU's

    :param n_gpu: str
        Maximum number of requested GPU's

    :param gpu_vendor: str
        Name of the GPU vendor
            -> amd: AMD
            -> nvidia: NVIDIA

    :param memory_request: str
        Memory request

    :param memory_limit: str
        Limit of the requested memory

    :param ephemeral_storage_request: str
        Ephemeral storage request (cloud based additional memory storage)

    :param ephemeral_storage_limit: str
        Limit of the requested ephemeral storage (cloud based additional memory storage)

    :param instance_name: str
        Name of the used AWS instance (value)

    :param max_cache_staleness: str
        Maximum of staleness days of the component cache

    :param kwargs: dict
        Key-word arguments for class ImageProcessor and compiling model configuration

    :return: dsl.ContainerOp
        Container operator for image classification
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-train_data_set_path', train_data_set_path,
                        '-test_data_set_path', test_data_set_path,
                        '-labels', labels,
                        '-s3_output_path_evaluation_train_data', s3_output_path_evaluation_train_data,
                        '-s3_output_path_evaluation_test_data', s3_output_path_evaluation_test_data,
                        '-s3_output_file_path_model_artifact', s3_output_file_path_model_artifact,
                        '-n_channels', n_channels,
                        '-image_height', image_height,
                        '-image_width', image_width,
                        '-learning_rate', learning_rate,
                        '-optimizer', optimizer,
                        '-initializer', initializer,
                        '-activation', activation,
                        '-batch_size', batch_size,
                        '-n_epoch', n_epoch,
                        '-n_conv_layers', n_conv_layers,
                        '-start_n_filters_conv_layers', start_n_filters_conv_layers,
                        '-max_n_filters_conv_layers', max_n_filters_conv_layers,
                        '-up_size_n_filters_period', up_size_n_filters_period,
                        '-dropout_rate_conv_layers', dropout_rate_conv_layers,
                        '-pool_size_conv_layers', pool_size_conv_layers,
                        '-checkpoint_epoch_interval', checkpoint_epoch_interval,
                        '-evaluation_epoch_interval', evaluation_epoch_interval,
                        '-print_model_architecture', int(print_model_architecture),
                        ]
    if val_data_set_path is not None:
        _arguments.extend(['-val_data_set_path', val_data_set_path])
    if s3_output_path_evaluation_val_data is not None:
        _arguments.extend(['-s3_output_path_evaluation_val_data', s3_output_path_evaluation_val_data])
    if kwargs is not None:
        _arguments.extend(['-kwargs', kwargs])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='image_classification',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             #file_outputs={'imp_features': output_path_imp_features},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name(display_name)
    add_container_op_parameters(container_op=_task,
                                n_cpu_request=n_cpu_request,
                                n_cpu_limit=n_cpu_limit,
                                n_gpu=n_gpu,
                                gpu_vendor=gpu_vendor,
                                memory_request=memory_request,
                                memory_limit=memory_limit,
                                ephemeral_storage_request=ephemeral_storage_request,
                                ephemeral_storage_limit=ephemeral_storage_limit,
                                instance_name=instance_name,
                                max_cache_staleness=max_cache_staleness
                                )
    return _task
