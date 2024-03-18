"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import os

from aws import filter_files_from_s3, load_file_from_s3, save_file_to_s3
from classifier import ImageClassifier
from custom_logger import Log
from typing import Dict, List


PARSER = argparse.ArgumentParser(description="image classification")
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='file path of the training image data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='file path of the test image data set')
PARSER.add_argument('-val_data_set_path', type=str, required=False, default=None, help='file path of the validation image data set')
PARSER.add_argument('-labels', nargs='+', required=True, default=None, help='class labels')
PARSER.add_argument('-n_channels', type=int, required=False, default=3, help='number of image channels to use')
PARSER.add_argument('-image_height', type=int, required=False, default=256, help='height of the images')
PARSER.add_argument('-image_width', type=int, required=False, default=256, help='width of the images')
PARSER.add_argument('-learning_rate', type=float, required=False, default=0.001, help='learning rate of the neural network optimizer component')
PARSER.add_argument('-optimizer', type=str, required=False, default='adam', help='name of the optimizer')
PARSER.add_argument('-initializer', type=str, required=False, default='he_normal', help='name of the initializer')
PARSER.add_argument('-activation', type=str, required=False, default='relu', help='name of the activation function used in convolutional layers')
PARSER.add_argument('-batch_size', type=int, required=False, default=32, help='batch size')
PARSER.add_argument('-n_epoch', type=int, required=False, default=300, help='number of epochs to train')
PARSER.add_argument('-n_conv_layers', type=int, required=False, default=7, help='number of convolutional layers')
PARSER.add_argument('-start_n_filters_conv_layers', type=int, required=False, default=32, help='number of filters used in first convolutional layer')
PARSER.add_argument('-max_n_filters_conv_layers', type=int, required=False, default=256, help='maximum number of filter used in all convolutional layers')
PARSER.add_argument('-dropout_rate_conv_layers', type=float, required=False, default=0.0, help='dropout rate used after each convolutional layer')
PARSER.add_argument('-up_size_n_filters_period', type=int, required=False, default=3, help='period to up-size (double) filters in convolutionaly layers')
PARSER.add_argument('-pool_size_conv_layers', type=int, required=False, default=2, help='pooling size of the max pooling layer')
PARSER.add_argument('-checkpoint_epoch_interval', type=int, required=False, default=5, help='epoch interval of saving model checkpoint')
PARSER.add_argument('-evaluation_epoch_interval', type=int, required=False, default=1, help='epoch interval of evaluating model')
PARSER.add_argument('-print_model_architecture', type=int, required=False, default=1, help='whether to print model architecture or not')
PARSER.add_argument('-kwargs', type=str, required=False, default=None, help='key-word arguments')
PARSER.add_argument('-s3_output_path_evaluation_train_data', type=str, required=True, default=None, help='complete file path of the evaluation training data output')
PARSER.add_argument('-s3_output_path_evaluation_test_data', type=str, required=True, default=None, help='complete file path of the evaluation test data output')
PARSER.add_argument('-s3_output_path_evaluation_val_data', type=str, required=False, default=None, help='complete file path of the evaluation validation data output')
PARSER.add_argument('-s3_output_file_path_model_artifact', type=str, required=True, default=None, help='S3 file path of the model artifact output')
ARGS = PARSER.parse_args()


def image_classification(train_data_set_path: str,
                         test_data_set_path: str,
                         labels: List[str],
                         s3_output_path_evaluation_train_data: str,
                         s3_output_path_evaluation_test_data: str,
                         s3_output_file_path_model_artifact: str,
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
                         **kwargs
                         ) -> None:
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

    :param kwargs: dict
        Key-word arguments for class ImageProcessor and compiling model configuration
    """
    _train_folder: str = train_data_set_path.split('/')[-1]
    if len(_train_folder) == 0:
        _train_folder = train_data_set_path.split('/')[-2]
    os.mkdir(_train_folder)
    _train_image_evaluation: Dict[str, list] = dict(file_path=[], label=[])
    for train_label in labels:
        _train_label_sub_folder: str = f'{_train_folder}/{train_label}'
        os.mkdir(_train_label_sub_folder)
        _train_images: List[str] = filter_files_from_s3(file_path=train_data_set_path, obj_ids=[train_label])
        Log().log(msg=f'Filter {len(_train_images)} elements from S3 bucket ({train_data_set_path})')
        for train_image in _train_images:
            if len(train_image.split('/')[-1]) == 0:
                continue
            _train_image_evaluation['file_path'].append(train_image)
            _train_image_evaluation['label'].append(train_label)
            load_file_from_s3(file_path=train_image, local_file_path=f'{_train_label_sub_folder}/{train_image.split("/")[-1]}')
        Log().log(msg=f'Load {len(_train_images)} images of label {train_label} for training')
    _test_folder: str = test_data_set_path.split('/')[-1]
    if len(_test_folder) == 0:
        _test_folder = test_data_set_path.split('/')[-2]
    os.mkdir(_test_folder)
    _test_image_evaluation: Dict[str, list] = dict(file_path=[], label=[])
    for test_label in labels:
        _test_label_sub_folder: str = f'{_test_folder}/{test_label}'
        os.mkdir(_test_label_sub_folder)
        _test_images: List[str] = filter_files_from_s3(file_path=test_data_set_path, obj_ids=[test_label])
        Log().log(msg=f'Filter {len(_test_images)} elements from S3 bucket ({test_data_set_path})')
        for test_image in _test_images:
            if len(test_image.split('/')[-1]) == 0:
                continue
            _test_image_evaluation['file_path'].append(test_image)
            _test_image_evaluation['label'].append(test_label)
            load_file_from_s3(file_path=test_image, local_file_path=f'{_test_label_sub_folder}/{test_image.split("/")[-1]}')
        Log().log(msg=f'Load {len(_test_images)} images of label {test_label} for validation')
    _image_classifier: ImageClassifier = ImageClassifier(file_path_train_images=_train_folder,
                                                         file_path_test_images=_test_folder,
                                                         labels=labels,
                                                         n_epoch=n_epoch,
                                                         n_channels=n_channels,
                                                         image_height=image_height,
                                                         image_width=image_width,
                                                         learning_rate=learning_rate,
                                                         optimizer=optimizer,
                                                         initializer=initializer,
                                                         activation=activation,
                                                         batch_size=batch_size,
                                                         n_conv_layers=n_conv_layers,
                                                         start_n_filters_conv_layers=start_n_filters_conv_layers,
                                                         max_n_filters_conv_layers=max_n_filters_conv_layers,
                                                         up_size_n_filters_period=up_size_n_filters_period,
                                                         dropout_rate_conv_layers=dropout_rate_conv_layers,
                                                         pool_size_conv_layers=(pool_size_conv_layers, pool_size_conv_layers),
                                                         checkpoint_epoch_interval=checkpoint_epoch_interval,
                                                         print_model_architecture=print_model_architecture,
                                                         **kwargs
                                                         )
    _image_classifier.train()
    _model_file_name: str = s3_output_file_path_model_artifact.split('/')[-1]
    _image_classifier.model.save(filepath=_model_file_name)
    save_file_to_s3(file_path=s3_output_file_path_model_artifact, obj=None, input_file_path=_model_file_name)
    Log().log(msg=f'Save model artifact: {s3_output_file_path_model_artifact}')


if __name__ == '__main__':
    if ARGS.labels:
        ARGS.labels = ast.literal_eval(ARGS.labels[0])
    if ARGS.kwargs:
        ARGS.kwargs = ast.literal_eval(ARGS.kwargs)
    image_classification(train_data_set_path=ARGS.train_data_set_path,
                         test_data_set_path=ARGS.test_data_set_path,
                         labels=ARGS.labels,
                         s3_output_path_evaluation_train_data=ARGS.s3_output_path_evaluation_train_data,
                         s3_output_path_evaluation_test_data=ARGS.s3_output_path_evaluation_test_data,
                         s3_output_file_path_model_artifact=ARGS.s3_output_file_path_model_artifact,
                         n_channels=ARGS.n_channels,
                         image_height=ARGS.image_height,
                         image_width=ARGS.image_width,
                         learning_rate=ARGS.learning_rate,
                         optimizer=ARGS.optimizer,
                         initializer=ARGS.initializer,
                         activation=ARGS.activation,
                         batch_size=ARGS.batch_size,
                         n_epoch=ARGS.n_epoch,
                         n_conv_layers=ARGS.n_conv_layers,
                         start_n_filters_conv_layers=ARGS.start_n_filters_conv_layers,
                         max_n_filters_conv_layers=ARGS.max_n_filters_conv_layers,
                         dropout_rate_conv_layers=ARGS.dropout_rate_conv_layers,
                         up_size_n_filters_period=ARGS.up_size_n_filters_period,
                         pool_size_conv_layers=ARGS.pool_size_conv_layers,
                         val_data_set_path=ARGS.val_data_set_path,
                         s3_output_path_evaluation_val_data=ARGS.s3_output_path_evaluation_val_data,
                         checkpoint_epoch_interval=ARGS.checkpoint_epoch_interval,
                         evaluation_epoch_interval=ARGS.evaluation_epoch_interval,
                         print_model_architecture=bool(ARGS.print_model_architecture),
                         **ARGS.kwargs
                         )
