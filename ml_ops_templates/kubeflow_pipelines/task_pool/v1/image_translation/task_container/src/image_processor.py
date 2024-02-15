"""

Image processor

"""

import cv2
import numpy as np
import os
import random

from aws import filter_files_from_s3, load_file_from_s3, save_file_to_s3
from custom_logger import Log
from glob import glob
from typing import List, Tuple


class ImageProcessor:
    """
    Class for processing, loading and saving document images
    """
    def __init__(self,
                 file_path_clean_images: str,
                 file_path_noisy_images: str,
                 file_path_multi_noisy_images: List[str] = None,
                 n_channels: int = 1,
                 batch_size: int = 1,
                 image_resolution: tuple = None,
                 normalize: bool = False,
                 flip: bool = True,
                 crop: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 file_type: str = None
                 ):
        """
        :param file_path_clean_images: str
            Complete file path of the clean images

        :param file_path_noisy_images: str
            Complete file path of the noisy images

        :param file_path_multi_noisy_images: List[str]
            Complete file paths of several noisy images

        :param n_channels: int
            Number of channels of the image
                -> 0: gray
                -> 3: color (rgb)

        :param batch_size: int
            Batch size

        :param image_resolution: tuple
            Force resizing image into given resolution

        :param normalize: bool
            Whether to normalize image (rescale to 0 - 1) or not

        :param flip: bool
            Whether to flip image based on probability distribution or not

        :parm crop: Tuple[Tuple[int, int], Tuple[int, int]]
            Define image cropping

        :param file_type: str
            Specific image file type
        """
        self.file_path_clean_images: str = file_path_clean_images
        self.file_path_noisy_images: str = file_path_noisy_images
        self.file_path_multi_noisy_images: List[str] = file_path_multi_noisy_images
        self.file_type: str = '' if file_type is None else file_type
        self.n_channels: int = 3 if n_channels > 1 else 1
        self.batch_size: int = batch_size
        self.n_batches: int = 0
        self.image_resolution: tuple = image_resolution
        self.normalize: bool = normalize
        self.flip: bool = flip
        self.crop: Tuple[Tuple[int, int], Tuple[int, int]] = crop

    @staticmethod
    def _de_normalize(images: np.array) -> np.array:
        """
        De-normalize images

        :param images: np.array
            Images
        """
        return images * 255

    @staticmethod
    def _normalize(images: np.array) -> np.array:
        """
        Normalize images

        :param images: np.array
            Images
        """
        return images / 255

    def _read_image(self, file_path: str) -> np.array:
        """
        Read image

        :param file_path: str
            Complete file path of the image to read

        :return np.array
            Image
        """
        _image: np.array = load_file_from_s3(file_path=file_path, image_channels=self.n_channels)
        if self.crop is not None:
            _image = _image[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
        if self.image_resolution is not None:
            _image = cv2.resize(src=_image, dsize=self.image_resolution, interpolation=cv2.INTER_AREA)
        if self.flip:
            if np.random.random() > 0.5:
                if np.random.random() > 0.5:
                    _direction: int = 0
                else:
                    _direction: int = 1
                _image = cv2.flip(src=_image, flipCode=_direction, dst=None)
        return _image

    def load_batch(self, label: int = -1) -> Tuple[np.array, np.array, List[int]]:
        """
        Load batch images for each group (clean & noisy) separately

        :param label: int
            Number of noise type to load (multi noise images only)

        :return Tuple[np.array, np.array, List[str]]
            Arrays of clean & noisy images as well as noise labels
        """
        #if self.file_path_multi_noisy_images is None:
        #    _label: int = 0
        #    _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        #else:
        #    if label >= 0:
        #        _label: int = label
        #    else:
        #        _label: int = random.randint(a=0, b=len(self.file_path_multi_noisy_images) - 1)
        #    _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_multi_noisy_images[_label], f'*{self.file_type}'))
        #_file_paths_clean: List[str] = glob(os.path.join('.', self.file_path_clean_images, f'*{self.file_type}'))
        _file_paths_A: List[str] = filter_files_from_s3(file_path=self.file_path_clean_images, obj_ids=['.jpg', '.jpeg', '.png'])
        _file_paths_B: List[str] = filter_files_from_s3(file_path=self.file_path_noisy_images, obj_ids=['.jpg', '.jpeg', '.png'])
        self.n_batches = int(min(len(_file_paths_A), len(_file_paths_B)) / self.batch_size)
        _total_samples: int = self.n_batches * self.batch_size
        _file_paths_clean_sample: List[str] = np.random.choice(_file_paths_A, _total_samples, replace=False)
        _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_B, _total_samples, replace=False)
        for i in range(0, self.n_batches - 1, 1):
            _batch_clean: List[str] = _file_paths_clean_sample[i * self.batch_size:(i + 1) * self.batch_size]
            _batch_noisy: List[str] = _file_paths_noisy_sample[i * self.batch_size:(i + 1) * self.batch_size]
            _images_clean, _images_noisy = [], []
            for path_image_clean, path_image_noisy in zip(_batch_clean, _batch_noisy):
                _images_clean.append(self._read_image(file_path=path_image_clean))
                _images_noisy.append(self._read_image(file_path=path_image_noisy))
            if self.normalize:
                yield self._normalize(np.array(_images_clean)), self._normalize(np.array(_images_noisy)), label
            else:
                yield np.array(_images_clean), np.array(_images_noisy), label

    def load_all_images(self) -> Tuple[np.array, np.array]:
        """
        Load all images without batching
        """
        if self.file_path_multi_noisy_images is None:
            _label: int = 0
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        else:
            _label: int = random.randint(a=0, b=len(self.file_path_multi_noisy_images) - 1)
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_multi_noisy_images[_label], f'*{self.file_type}'))
        _file_paths_clean: List[str] = glob(os.path.join('.', self.file_path_clean_images, f'*{self.file_type}'))
        sorted(_file_paths_clean, reverse=False)
        sorted(_file_paths_noisy, reverse=False)
        _images_clean, _images_noisy = [], []
        for path_image_clean, path_image_noisy in zip(_file_paths_clean, _file_paths_noisy):
            _images_clean.append(self._read_image(file_path=path_image_clean))
            _images_noisy.append(self._read_image(file_path=path_image_noisy))
        if self.normalize:
            return self._normalize(np.array(_images_clean)), self._normalize(np.array(_images_noisy))
        else:
            return np.array(_images_clean), np.array(_images_noisy)

    def load_images(self, n_images: int = None, label: int = None) -> Tuple[str, np.array]:
        """
        Load images without batching

        :param n_images: int
            Number of images

        :param label: int
            Label number

        :return Tuple[List[str], np.array]
            List of image file names & array of loaded images
        """
        _images_path: List[str] = []
        _images_noisy: List[np.array] = []
        if self.file_path_multi_noisy_images is None:
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        else:
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_multi_noisy_images[label], f'*{self.file_type}'))
        if n_images is not None and n_images > 0:
            for i in range(0, n_images, 1):
                _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, 1, replace=False)
                for path_image_noisy in _file_paths_noisy_sample:
                    _images_path.append(path_image_noisy)
                    _images_noisy.append(self._read_image(file_path=path_image_noisy))
                if self.normalize:
                    yield _images_path, self._normalize(np.array(_images_noisy))
                else:
                    yield _images_path, np.array(_images_noisy)
        else:
            self.n_batches = int(len(_file_paths_noisy))
            _total_samples: int = self.n_batches * self.batch_size
            _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, _total_samples, replace=False)
            for i in range(0, self.n_batches - 1, 1):
                _batch_noisy: List[str] = _file_paths_noisy_sample[i * self.batch_size:(i + 1) * self.batch_size]
                for path_image_noisy in _batch_noisy:
                    _images_path.append(path_image_noisy)
                    _images_noisy.append(self._read_image(file_path=path_image_noisy))
                if self.normalize:
                    yield _images_path, self._normalize(np.array(_images_noisy))
                else:
                    yield _images_path, np.array(_images_noisy)

    def save_image(self, image: np.array, output_file_path: str):
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
        Log().log(msg=f'Save image: {output_file_path}')
