""" Module including dithering algorithms """
import argparse
import os

from PIL import Image as PillowImage
from imgprocalgs.algorithms.utilities import Image, get_greyscale
from .base import BaseImageAlgorithm


class FloydSteinberg(BaseImageAlgorithm):
    """
    Floyd Steinberg algorithm used for reducing grayscale image to black and white.
    Source: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
    """

    BLACK = "#000000"
    WHITE = "#FFFFFF"

    def __init__(self, image_path: str, destination_path: str, factor: int):
        super().__init__(image_path, destination_path)

        self.min_factor = 0
        self.max_factor = int(get_greyscale(255, 255, 255))  # max greyscale value

        if not self.min_factor < factor < self.max_factor:
            raise ValueError(f"Factor value should be from 0 to {self.max_factor}")

        self.image = Image(image_path)
        self.factor = factor

        # raw error table and output image
        self.width, self.height = self.image.get_size()
        self.greyscale_image = self.image2greyscale()

        # error size + 1 than input image because of lack of if statements
        self.error_table = [[0 for _ in range(self.height + 2)] for __ in range(self.width + 2)]

        self.output_image = PillowImage.new("RGB", (self.width, self.height), self.WHITE)
        self.output_pixels = self.output_image.load()
        self.output_image_name = "output_floyd_steinberg.jpg"

    def output_filename(self) -> str:
        return self.output_image_name

    def image2greyscale(self):
        greyscale_image = PillowImage.new("RGB", (self.width, self.height), self.WHITE)
        pixels = greyscale_image.load()

        for x in range(self.width):
            for y in range(self.height):
                grey_value = int(get_greyscale(*self.image.pixels[x, y]))
                pixels[x, y] = (grey_value, grey_value, grey_value)

        greyscale_image.save(os.path.join(self.destination_path, "output_greyscale.jpg"))
        return greyscale_image

    def process(self):
        input_pixels = self.greyscale_image.load()

        for x in range(self.width):
            for y in range(self.height):
                if self.factor > input_pixels[x, y][0] + self.error_table[x][y]:
                    self.output_pixels[x, y] = (0, 0, 0)
                    current_error = input_pixels[x, y][0] + self.error_table[x][y]
                else:
                    self.output_pixels[x, y] = (255, 255, 255)
                    current_error = input_pixels[x, y][0] + self.error_table[x][y] - 255

                self._propagate_error(x, y, current_error)

        self.save()

    def _propagate_error(self, x, y, current_error):
        self.error_table[x + 1][y] += 7 / 16 * current_error
        self.error_table[x + 1][y + 1] += 3 / 16 * current_error
        self.error_table[x][y + 1] += 5 / 16 * current_error
        self.error_table[x - 1][y + 1] += 1 / 16 * current_error


class JarvisJudiceNinke(FloydSteinberg):
    """
    Floyd Steinberg extension
    https://en.wikipedia.org/wiki/Dither
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_table = [[0 for _ in range(self.height + 3)] for __ in range(self.width + 3)]
        self.output_image_name = "output_jarvis_judice_ninke.jpg"


class Stucki(FloydSteinberg):
    """
    Floyd Steinberg extension
    https://en.wikipedia.org/wiki/Dither
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_table = [[0 for _ in range(self.height + 3)] for __ in range(self.width + 3)]
        self.output_image_name = "output_stucki.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Dithering algorithm")
    parser.add_argument("--src", type=str, help="Source file path.")
    parser.add_argument("--dest", type=str, help="Destination file path.", default="data/")
    parser.add_argument("--method", type=str, help="Dithering method")
    parser.add_argument("--factor", type=int, help="Factor when goes black/white pixel")
    return parser.parse_args()


def main():
    name2alg = {
        "floydsteinberg": FloydSteinberg,
        "jarvisjudiceninke": JarvisJudiceNinke,
        "stucki": Stucki
    }

    args = parse_args()
    cls = name2alg.get(args.method.lower(), None)

    if not cls:
        raise Exception(f"No such dithering algorithm: {args.method}")

    cls(args.src, args.dest, args.factor).process()
