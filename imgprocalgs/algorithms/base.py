""" Module including base class for class based algorithms"""
from abc import abc, ABC, abstractmethod
from imgprocalgs.algorithms.utilities import Image, create_empty_image


class BaseImageAlgorithm(ABC):
    """
    Base class for all image processing algorithms
    """

    def __init__(self, image_path: str, destination_path: str):
        self.image_path = image_path
        self.destination_path = destination_path

        self.input_image = Image(image_path)
        self.width, self.height = self.input_image.get_size()
        self.pixels = self.input_image.pixels

        self.output_image = create_empty_image(self.width, self.height)
        self.output_pixels = self.output_image.load()

    @abstractmethod
    def process(self):
        """
        Core algorithm logic
        """
        pass

    @abstractmethod
    def output_filename(self) -> str:
        """
        Output image filename
        """
        pass

    def save(self):
        self.output_image.save(
            f"{self.destination_path}/{self.output_filename()}"
        )
