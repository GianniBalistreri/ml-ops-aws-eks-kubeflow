"""

Task: ... (Function to run in container)

"""

import argparse

from auto_encoder import AutoEncoder
from custom_logger import Log
from cycle_gan import CycleGAN


PARSER = argparse.ArgumentParser(description="image translation")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-missing_value_threshold', type=float, required=False, default=0.95, help='threshold to classify features as invalid based on the amount of missing values')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_data_missing_data', type=str, required=True, default=None, help='file path of features containing too much missing data output')
PARSER.add_argument('-output_file_path_invariant', type=str, required=True, default=None, help='file path of invariant features output')
PARSER.add_argument('-output_file_path_duplicated', type=str, required=True, default=None, help='file path of duplicated features output')
PARSER.add_argument('-output_file_path_valid_features', type=str, required=True, default=None, help='file path of valid features output')
PARSER.add_argument('-output_file_path_prop_valid_features', type=str, required=True, default=None, help='file path of the proportion of valid features output')
PARSER.add_argument('-s3_output_file_path_data_health_check', type=str, required=False, default=None, help='S3 file path of the data health check output')
ARGS = PARSER.parse_args()


def image_translation(unpaired: bool,
                      train_data_set_path: str,
                      test_data_set_path: str = None,
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
                      include_moe_layers: bool = False,
                      start_n_filters_moe_embedder: int = 32,
                      n_conv_layers_moe_embedder: int = 7,
                      max_n_filters_embedder: int = 64,
                      dropout_rate_moe_embedder: float = 0.0,
                      n_embedding_features: int = 64,
                      gate_after_each_conv_res_net_block: bool = True,
                      n_hidden_layers_moe_fc_gated_net: int = 1,
                      n_hidden_layers_moe_fc_classifier: int = 1,
                      dropout_rate_moe_fc_gated_net: float = 0.0,
                      n_noise_types_moe_fc_classifier: int = 4,
                      dropout_rate_moe_fc_classifier: float = 0.0,
                      print_model_architecture: bool = True,
                      s3_output_file_path_model_artifact: str = None
                      ) -> None:
    """
    Translate paired or unpaired image characteristics

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

    :param include_moe_layers: bool
        Whether to use mixture of experts layers in residual network architecture

    :param start_n_filters_moe_embedder: int
        Number of filters used in first convolutional layer in embedding network

    :param n_conv_layers_moe_embedder: int
        Number of convolutional layers in discriminator network

    :param max_n_filters_embedder: int
        Maximum number of filters in embedder

    :param dropout_rate_moe_embedder: float
        Dropout rate used after each convolutional layer in mixture of experts embedder network

    :param n_embedding_features: int
        Number of embedding output features

    :param gate_after_each_conv_res_net_block: bool
        Whether to use gated network after each convolutional layer in residual network block or just in the end

    :param n_hidden_layers_moe_fc_gated_net: int
        Number of hidden layers of the fully connected gated network used to process mixture of experts embedding output

    :param n_hidden_layers_moe_fc_classifier: int
        Number of hidden layers of the fully connected gated network used to classify mixture of experts embedding output (noise type)

    :param dropout_rate_moe_fc_gated_net: float
        Dropout rate used after each convolutional layer in mixture of experts fully connected gated network

    :param n_noise_types_moe_fc_classifier: int
        Number of classes (noise types) to classify

    :param dropout_rate_moe_fc_classifier: float
        Dropout rate used after each convolutional layer in mixture of experts fully connected classification network

    :param print_model_architecture: bool
        Whether to print architecture of cycle-gan model components (discriminators & generators) or not

    :param kwargs: dict
        Key-word arguments for class ImageProcessor and compiling model configuration
    """
    if unpaired:
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=None,
                                        file_path_train_noisy_images=None,
                                        file_path_eval_noisy_images=None,
                                        file_path_moe_noisy_images=None,
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
                                        include_moe_layers=include_moe_layers,
                                        start_n_filters_moe_embedder=start_n_filters_moe_embedder,
                                        n_conv_layers_moe_embedder=n_conv_layers_moe_embedder,
                                        max_n_filters_embedder=max_n_filters_embedder,
                                        dropout_rate_moe_embedder=dropout_rate_moe_embedder,
                                        n_embedding_features=n_embedding_features,
                                        gate_after_each_conv_res_net_block=gate_after_each_conv_res_net_block,
                                        n_hidden_layers_moe_fc_gated_net=n_hidden_layers_moe_fc_gated_net,
                                        n_hidden_layers_moe_fc_classifier=n_hidden_layers_moe_fc_classifier,
                                        dropout_rate_moe_fc_gated_net=dropout_rate_moe_fc_gated_net,
                                        n_noise_types_moe_fc_classifier=n_noise_types_moe_fc_classifier,
                                        dropout_rate_moe_fc_classifier=dropout_rate_moe_fc_classifier,
                                        print_model_architecture=print_model_architecture
                                        )
        _cycle_gan.train(model_output_path=s3_output_file_path_model_artifact,
                         n_epoch=300,
                         asynchron=False,
                         discriminator_batch_size=100,
                         generator_batch_size=10,
                         early_stopping_batch=0,
                         random_noise_types=False,
                         checkpoint_epoch_interval=5,
                         evaluation_epoch_interval=1
                         )
    else:
        _auto_encoder: AutoEncoder = AutoEncoder(file_path_train_clean_images='',
                                                 file_path_train_noisy_images='',
                                                 n_channels=n_channels,
                                                 image_height=image_height,
                                                 image_width=image_width,
                                                 learning_rate=learning_rate,
                                                 optimizer=optimizer,
                                                 initializer=initializer,
                                                 batch_size=batch_size,
                                                 start_n_filters=64,
                                                 n_conv_layers=3,
                                                 dropout_rate=0.0,
                                                 print_model_architecture=print_model_architecture
                                                 )
        _auto_encoder.train(model_output_path=s3_output_file_path_model_artifact,
                            n_epoch=30,
                            early_stopping_patience=0
                            )


if __name__ == '__main__':
    data_health_check(data_set_path=ARGS.data_set_path,
                      analytical_data_types=ARGS.analytical_data_types,
                      output_file_path_missing_data=ARGS.output_file_path_missing_data,
                      output_file_path_invariant=ARGS.output_file_path_invariant,
                      output_file_path_duplicated=ARGS.output_file_path_duplicated,
                      output_file_path_valid_features=ARGS.output_file_path_valid_features,
                      output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
                      missing_value_threshold=ARGS.missing_value_threshold,
                      sep=ARGS.sep,
                      s3_output_file_path_data_health_check=ARGS.s3_output_file_path_data_health_check
                      )
