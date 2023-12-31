"""

Task: ... (Function to run in container)

"""

import argparse

from image_processor import ImageProcessor
from typing import List, Tuple


PARSER = argparse.ArgumentParser(description="image processing")
PARSER.add_argument('-file_paths', type=list, required=True, default=None, help='file paths of the images')
PARSER.add_argument('-n_channels', type=int, required=False, default=1, help='number of image channels')
PARSER.add_argument('-image_resolution', type=tuple, required=False, default=None, help='image resolution')
PARSER.add_argument('-normalize', type=int, required=False, default=0, help='whether to normalize image data or not')
PARSER.add_argument('-flip', type=int, required=False, default=1, help='whether to flip image data or not')
PARSER.add_argument('-crop', type=tuple, required=False, default=None, help='image cropping configuration')
PARSER.add_argument('-skew', type=int, required=False, default=0, help='whether to skew image data or not')
PARSER.add_argument('-deskew', type=int, required=False, default=0, help='whether to deskew image data or not')
PARSER.add_argument('-denoise_by_blurring', type=int, required=False, default=0, help='whether to denoise image data by blurring or not')
PARSER.add_argument('-generate_noise_blur', type=int, required=False, default=0, help='whether to generate blur noise in image data or not')
PARSER.add_argument('-blur_type', type=str, required=False, default='average', help='blur type')
PARSER.add_argument('-blur_kernel_size', type=tuple, required=False, default=(9, 9), help='kernel size for blurring')
PARSER.add_argument('-generate_noise_fade', type=int, required=False, default=0, help='whether to generate fade noise in image data or not')
PARSER.add_argument('-fade_brightness', type=int, required=False, default=20, help='brightness of the fade noise')
PARSER.add_argument('-generate_noise_salt_pepper', type=int, required=False, default=0, help='whether to generate salt & pepper noise in image data or not')
PARSER.add_argument('-salt_pepper_number_of_pixel_edges', type=tuple, required=False, default=(100000, 500000), help='number of pixel edges used to generate salt & pepper noise')
PARSER.add_argument('-generate_noise_watermark', type=int, required=False, default=0, help='whether to generate watermark noise in image data or not')
PARSER.add_argument('-watermark_files', type=list, required=False, default=None, help='complete file paths of the watermark noise')
PARSER.add_argument('-save_as_image', type=int, required=False, default=1, help='whether to save image data as images or as a aggregated binary file')
PARSER.add_argument('-s3_output_image_path', type=str, required=False, default=None, help='S3 path of the image output')
PARSER.add_argument('-s3_output_file_path_array', type=str, required=False, default=None, help='S3 file path of the aggregated binary output')
ARGS = PARSER.parse_args()


def image_processor(file_paths: List[str],
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
                    save_as_image: bool = True,
                    s3_output_image_path: str = None,
                    s3_output_file_path_array: str = None
                    ) -> None:
    """
    Process images

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

    :param save_as_image: bool
            Whether to save processed image as image or multi-dimensional array

    :param s3_output_image_path: str
        Path of the image output

    :param s3_output_file_path_array: str
        Complete file path of the processed image array output
    """
    _image_processor: ImageProcessor = ImageProcessor(file_paths=file_paths,
                                                      n_channels=n_channels,
                                                      image_resolution=image_resolution,
                                                      normalize=normalize,
                                                      flip=flip,
                                                      crop=crop,
                                                      skew=skew,
                                                      deskew=deskew,
                                                      denoise_by_blurring=denoise_by_blurring,
                                                      generate_noise_blur=generate_noise_blur,
                                                      blur_type=blur_type,
                                                      blur_kernel_size=blur_kernel_size,
                                                      generate_noise_fade=generate_noise_fade,
                                                      fade_brightness=fade_brightness,
                                                      generate_noise_salt_pepper=generate_noise_salt_pepper,
                                                      salt_pepper_number_of_pixel_edges=salt_pepper_number_of_pixel_edges,
                                                      generate_noise_watermark=generate_noise_watermark,
                                                      watermark_files=watermark_files,
                                                      file_type=None
                                                      )
    _image_processor.main(save_as_image=save_as_image,
                          image_path=s3_output_image_path,
                          file_path_array=s3_output_file_path_array
                          )


if __name__ == '__main__':
    image_processor(file_paths=ARGS.file_paths,
                    n_channels=ARGS.n_channels,
                    image_resolution=ARGS.image_resolution,
                    normalize=ARGS.normalize,
                    flip=ARGS.flip,
                    crop=ARGS.crop,
                    skew=ARGS.skew,
                    deskew=ARGS.deskew,
                    denoise_by_blurring=ARGS.denoise_by_blurring,
                    generate_noise_blur=ARGS.generate_noise_blur,
                    blur_type=ARGS.blur_type,
                    blur_kernel_size=ARGS.blur_kernel_size,
                    generate_noise_fade=ARGS.generate_noise_fade,
                    fade_brightness=ARGS.fade_brightness,
                    generate_noise_salt_pepper=ARGS.generate_noise_salt_pepper,
                    salt_pepper_number_of_pixel_edges=ARGS.salt_pepper_number_of_pixel_edges,
                    generate_noise_watermark=ARGS.generate_noise_watermark,
                    watermark_files=ARGS.watermark_files,
                    save_as_image=ARGS.save_as_image,
                    s3_output_image_path=ARGS.s3_output_image_path,
                    s3_output_file_path_array=ARGS.s3_output_file_path_array
                    )
