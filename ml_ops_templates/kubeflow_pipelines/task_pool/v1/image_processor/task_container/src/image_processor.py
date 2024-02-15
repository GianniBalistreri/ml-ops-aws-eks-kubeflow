"""
Utility functions
"""

import cv2
import json
import numpy as np
import os
import random
import struct

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from glob import glob
from skimage.transform import resize
from typing import List, Tuple


class ImageProcessorException(Exception):
    """
    Class for handling exceptions for class ImageProcessor
    """
    pass


class ImageProcessor:
    """
    Class for processing images
    """
    def __init__(self,
                 file_paths: List[str],
                 n_channels: int = 1,
                 image_resolution: Tuple[int, int] = None,
                 normalize: bool = False,
                 flip: bool = True,
                 crop: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 skew: bool = False,
                 deskew: bool = False,
                 denoise_by_blurring: bool = False,
                 generate_noise_blur: bool = False,
                 blur_type: str = 'average',
                 blur_kernel_size: Tuple[int, int] = (9, 9),
                 generate_noise_fade: bool = False,
                 fade_brightness: int = 20,
                 generate_noise_salt_pepper: bool = False,
                 salt_pepper_number_of_pixel_edges: Tuple[int, int] = (100000, 500000),
                 generate_noise_watermark: bool = False,
                 watermark_files: List[str] = None,
                 file_type: str = None,
                 ):
        """
        :param file_paths: str
            Complete file path of the images

        :param n_channels: int
            Number of channels of the image
                -> 0: gray
                -> 3: color (rgb)

        :param image_resolution: Tuple[int, int]
            Force resizing image into given resolution

        :param normalize: bool
            Whether to normalize image (rescale to 0 - 1) or not

        :param flip: bool
            Whether to flip image based on probability distribution or not

        :param crop: Tuple[Tuple[int, int], Tuple[int, int]]
            Define image cropping

        :param skew: bool
            Whether to skew image or not

        :param deskew: bool
            Whether to deskew image or not

        :param denoise_by_blurring: bool
            Whether to skew image or not

        :param generate_noise_blur: bool
            Whether to generate blur noise or not

        :param blur_type: str
            Blur type name
                -> average: Average
                -> median: Median

        :param blur_kernel_size: Tuple[int, int]
            Kernel size

        :param generate_noise_fade: bool
            Whether to generate fade noise or not

        :param fade_brightness: int
            Desired brightness change

        :param generate_noise_salt_pepper: bool
            Whether to generate salt and pepper noise or not

        :param salt_pepper_number_of_pixel_edges: tuple
            Edges to draw number of pixels to coloring into white and black

        :param generate_noise_watermark: bool
            Whether to generate watermark noise or not

        :param watermark_files: List[str]
            Complete file path of the files containing watermarks

        :param file_type: str
            Specific image file type
        """
        self.file_paths: List[str] = file_paths
        self.file_type: str = '' if file_type is None else file_type
        self.n_channels: int = 3 if n_channels > 1 else 1
        self.image_resolution: Tuple[int, int] = image_resolution
        self.normalize: bool = normalize
        self.flip: bool = flip
        self.crop: Tuple[Tuple[int, int], Tuple[int, int]] = crop
        self.skew: bool = skew
        self.deskew: bool = deskew
        self.denoise_by_blurring: bool = denoise_by_blurring
        self.generate_noise_blur: bool = generate_noise_blur
        self.blur_type: str = blur_type if blur_type in ['average', 'median'] else 'average'
        self.blur_kernel_size: Tuple[int, int] = blur_kernel_size
        self.generate_noise_fade: bool = generate_noise_fade
        self.fade_brightness: int = fade_brightness if fade_brightness > 0 else 150
        self.generate_noise_salt_pepper: bool = generate_noise_salt_pepper
        self.salt_pepper_number_of_pixel_edges: Tuple[int, int] = salt_pepper_number_of_pixel_edges
        self.generate_noise_watermark: bool = generate_noise_watermark
        self.watermark_files: List[str] = watermark_files

    def _blur(self, image: np.ndarray) -> np.ndarray:
        """
        Generate blurred noise images

        :param image: np.ndarray
            Image data

        :return np.ndarray
            Skewed image data
        """
        if self.blur_type == 'average':
            _image = cv2.blur(image, self.blur_kernel_size)
        else:
            _image = cv2.medianBlur(image, self.blur_kernel_size[0])
        if self.image_resolution is not None:
            _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
        return _image

    @staticmethod
    def _de_normalize(images: np.array) -> np.array:
        """
        De-normalize images

        :param images: np.array
            Images
        """
        return images * 255

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image

        :param image: np.ndarray
            Image data

        :return np.ndarray
            Deskewed image data
        """
        _angle: float = self._get_angle(image=image)
        return self._rotate_image(image=image, angle=-1.0 * _angle)

    def _fade(self, image: np.ndarray) -> np.ndarray:
        """
        Generate faded noise images

        :param image: np.ndarray
            Image data

        :return np.ndarray
            Skewed image data
        """
        _hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _h, _s, _v = cv2.split(_hsv)
        _lim: float = 255 - self.fade_brightness
        _v[_v > _lim] = 255
        _v[_v <= _lim] += self.fade_brightness
        _final_hsv = cv2.merge((_h, _s, _v))
        _image = cv2.cvtColor(_final_hsv, cv2.COLOR_HSV2BGR)
        if self.image_resolution is not None:
            _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
        return _image

    def _get_angle(self, image: np.array) -> float:
        """
        Get angle of image

        :param image: np.array
            Image

        :return float
            Angle of the text in document image
        """
        if self.denoise_by_blurring:
            _image: np.array = cv2.GaussianBlur(src=image, ksize=(9, 9), sigmaX=0, dst=None, sigmaY=None, borderType=None)
        else:
            _image: np.array = image
        _image = cv2.threshold(src=_image,
                               thresh=0,
                               maxval=255,
                               type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                               dst=None
                               )[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        _kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(30, 5), anchor=None)
        _image = cv2.dilate(src=_image, kernel=_kernel, iterations=5, borderType=None, borderValue=None)
        # Find all contours
        _contours, _hierarchy = cv2.findContours(image=_image,
                                                 mode=cv2.RETR_LIST,
                                                 method=cv2.CHAIN_APPROX_SIMPLE,
                                                 contours=None,
                                                 hierarchy=None,
                                                 offset=None
                                                 )
        _contours = sorted(_contours, key=cv2.contourArea, reverse=True)
        # Find largest contour and surround in min area box
        _largest_contour: float = _contours[0]
        _min_area_rectangle: tuple = cv2.minAreaRect(points=_largest_contour)
        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        _angle: float = _min_area_rectangle[-1]
        if _angle < -45:
            _angle = 90 + _angle
        return -1.0 * _angle

    @staticmethod
    def _normalize(image: np.array) -> np.array:
        """
        Normalize image

        :param image: np.array
            Image data
        """
        return image / 255

    def _read_image(self, file_path: str) -> np.array:
        """
        Read image

        :param file_path: str
            Complete file path of the image to read

        :return np.array
            Image
        """
        return load_file_from_s3(file_path=file_path, image_channels=self.n_channels)

    @staticmethod
    def _rotate_image(image: np.array, angle: float) -> np.array:
        """
        Rotate image

        :param image: np.array
            Image

        :param angle: float
            Angle for rotation

        :return np.array
            Rotated images
        """
        _rotated_image: np.array = image
        (_height, _width) = _rotated_image.shape[:2]
        _center: tuple = (_width // 2, _height // 2)
        _rotation_matrix = cv2.getRotationMatrix2D(center=_center, angle=angle, scale=1.0)
        return cv2.warpAffine(src=_rotated_image,
                              M=_rotation_matrix,
                              dsize=(_width, _height),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE
                              )

    def _save_image(self, image: np.array, output_file_path: str):
        """
        Save image

        :param image: np.array
            Image to save

        :param output_file_path: str
            Complete file path of the output image
        """
        if self.normalize:
            _image: np.array = self._de_normalize(images=image)
        else:
            _image: np.array = image
        save_file_to_s3(file_path=output_file_path, obj=_image)

    def _salt_pepper(self, image: np.ndarray) -> np.ndarray:
        """
        Generate salt & pepper noise images

        :param image: np.ndarray
            Image data

        :return np.ndarray
            Skewed image data
        """
        _image: np.ndarray = image
        _height, _width, _ = _image.shape
        _number_of_pixels: int = random.randint(self.salt_pepper_number_of_pixel_edges[0],
                                                self.salt_pepper_number_of_pixel_edges[1]
                                                )
        # White pixels:
        for i in range(0, _number_of_pixels, 1):
            _y_coord = random.randint(0, _height - 1)
            _x_coord = random.randint(0, _width - 1)
            _image[_y_coord][_x_coord] = 255
        # Black pixels:
        for i in range(0, _number_of_pixels, 1):
            _y_coord = random.randint(0, _height - 1)
            _x_coord = random.randint(0, _width - 1)
            _image[_y_coord][_x_coord] = 0
        if self.image_resolution is not None:
            _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
        return _image

    def _skew(self, image: np.ndarray) -> np.ndarray:
        """
        Skew image

        :param image: np.ndarray
            Image data

        :return np.ndarray
            Skewed image data
        """
        _angle: float = self._get_angle(image=image)
        return self._rotate_image(image=image, angle=_angle)

    def _watermark(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate watermarked noise images
        """
        _watermarked_images: List[np.ndarray] = []
        _image: np.ndarray = image
        _height_image, _width_image, _ = _image.shape
        _center_y: int = int(_height_image / 2)
        _center_x: int = int(_width_image / 2)
        for watermark in self.watermark_files:
            _watermark = self._read_image(file_path=watermark)
            _height_watermark, _width_watermark, _ = _watermark.shape
            _top_y: int = _center_y - int(_height_watermark / 2)
            _left_x: int = _center_x - int(_width_watermark / 2)
            _bottom_y: int = _top_y + _height_watermark
            _right_x: int = _left_x + _width_watermark
            _destination = _image[_top_y:_bottom_y, _left_x:_right_x]
            _result = cv2.addWeighted(_destination, 1, _watermark, 0.5, 0)
            _image[_top_y:_bottom_y, _left_x:_right_x] = _result
            if self.image_resolution is not None:
                _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
            _watermarked_images.append(_image)
        return _watermarked_images

    def main(self, save_as_image: bool = True, image_path: str = None, file_path_array: str = None) -> None:
        """
        Image processing

        :param save_as_image: bool
            Whether to save processed image as image or multi-dimensional array

        :param image_path: str
            Path of the image output

        :param file_path_array: str
            Complete file path of the processed image array output
        """
        if image_path is None and file_path_array is None:
            raise ImageProcessorException('No output path found')
        _images: List[np.ndarray] = []
        for image_file_path in self.file_paths:
            _image: np.ndarray = self._read_image(file_path=image_file_path)
            if self.crop is not None:
                _image = _image[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
            if self.image_resolution is not None:
                _image = resize(image=_image, output_shape=self.image_resolution)
            if self.flip:
                if np.random.random() > 0.5:
                    if np.random.random() > 0.5:
                        _direction: int = 0
                    else:
                        _direction: int = 1
                    _image = cv2.flip(src=_image, flipCode=_direction, dst=None)
            if self.skew:
                _image = self._skew(image=_image)
            if self.deskew:
                _image = self._deskew(image=_image)
            if self.generate_noise_blur:
                _image = self._blur(image=_image)
            if self.generate_noise_fade:
                _image = self._fade(image=_image)
            if self.generate_noise_salt_pepper:
                _image = self._salt_pepper(image=_image)
            if self.generate_noise_watermark:
                _image = self._watermark(image=_image)
            if self.normalize:
                _image = self._normalize(image=_image)
            if save_as_image:
                _file_name: str = os.path.join(image_path, image_file_path.split('/')[-1])
                save_file_to_s3(file_path=_file_name, obj=_image)
                Log().log(msg=f'Save image: {_file_name}')
            else:
                _images.append(_image)
        if len(_images) > 0:
            save_file_to_s3(file_path=file_path_array, obj=_images)
            Log().log(msg=f'Save images: {file_path_array}')
