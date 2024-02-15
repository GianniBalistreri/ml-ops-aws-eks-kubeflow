"""

Kubeflow pipeline component: Image translation

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl


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
                      aws_account_id: str = '711117404296',
                      docker_image_name: str = 'ml-ops-generative-adversarial-neural-networks',
                      docker_image_tag: str = 'v1',
                      volume: dsl.VolumeOp = None,
                      volume_dir: str = '/mnt',
                      display_name: str = 'Image Translation',
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

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :param aws_account_id: str
        AWS account id

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
        Container operator for image translation
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-train_data_set_a_path', train_data_set_a_path,
                        '-train_data_set_b_path', train_data_set_b_path,
                        '-test_data_set_b_path', test_data_set_b_path,
                        '-s3_output_file_path_evaluation', s3_output_file_path_evaluation,
                        '-unpaired', int(unpaired),
                        '-n_channels', n_channels,
                        '-image_height', image_height,
                        '-image_width', image_width,
                        '-learning_rate', learning_rate,
                        '-optimizer', optimizer,
                        '-initializer', initializer,
                        '-batch_size', batch_size,
                        '-n_epoch', n_epoch,
                        '-start_n_filters_auto_encoder', start_n_filters_auto_encoder,
                        '-n_conv_layers_auto_encoder', n_conv_layers_auto_encoder,
                        '-dropout_rate_auto_encoder', dropout_rate_auto_encoder,
                        '-start_n_filters_discriminator', start_n_filters_discriminator,
                        '-max_n_filters_discriminator', max_n_filters_discriminator,
                        '-n_conv_layers_discriminator', n_conv_layers_discriminator,
                        '-dropout_rate_discriminator', dropout_rate_discriminator,
                        '-start_n_filters_generator', start_n_filters_generator,
                        '-max_n_filters_generator', max_n_filters_generator,
                        '-up_sample_n_filters_period', up_sample_n_filters_period,
                        '-generator_type', generator_type,
                        '-n_res_net_blocks', n_res_net_blocks,
                        '-n_conv_layers_generator_res_net', n_conv_layers_generator_res_net,
                        '-n_conv_layers_generator_u_net', n_conv_layers_generator_u_net,
                        '-dropout_rate_generator_res_net', dropout_rate_generator_res_net,
                        '-dropout_rate_generator_down_sampling', dropout_rate_generator_down_sampling,
                        '-dropout_rate_generator_up_sampling', dropout_rate_generator_up_sampling,
                        '-asynchron', int(asynchron),
                        '-discriminator_batch_size', discriminator_batch_size,
                        '-generator_batch_size', generator_batch_size,
                        '-early_stopping_batch', early_stopping_batch,
                        '-checkpoint_epoch_interval', checkpoint_epoch_interval,
                        '-evaluation_epoch_interval', evaluation_epoch_interval,
                        '-print_model_architecture', int(print_model_architecture),
                        ]
    if s3_output_file_path_model_artifact is not None:
        _arguments.extend(['-s3_output_file_path_model_artifact', s3_output_file_path_model_artifact])
    if kwargs is not None:
        _arguments.extend(['-kwargs', kwargs])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='image_translation',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
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
