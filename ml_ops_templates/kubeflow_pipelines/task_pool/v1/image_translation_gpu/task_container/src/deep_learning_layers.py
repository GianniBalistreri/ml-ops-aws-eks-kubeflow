"""

Additional neural network layers (keras / tensorflow)

"""

from tensorflow import pad
from tensorflow.keras.layers import Layer


class ConstantPadding2D(Layer):
    """
    Class for building constant padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), constant: int = 0, **kwargs):
        """
        :param padding: tuple
            Padding size

        :param constant: int
            Constant value

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply constant padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], mode='CONSTANT', constant_values=self.constant)


class ReflectionPadding2D(Layer):
    """
    Class for building reflection padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), **kwargs):
        """
        :param padding: tuple
            Padding size

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply reflection padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], 'REFLECT')


class ReplicationPadding2D(Layer):
    """
    Class for building replication padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), **kwargs):
        """
        :param padding: tuple
            Padding size

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply replication padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], 'SYMMETRIC')
