import os
from PIL import Image
from .base import BaseImageAlgorithm
import google.generativeai as genai 

class GenAIImageEnhancement(BaseImageAlgorithm):
    """
    GenAI-based image enhancement using Gemini's image_edit capabilities.
    """

    def __init__(
        self,
        image_path: str,
        destination_path: str,
        user_prompt: str = None
    ):
        super().__init__(image_path, destination_path)
        # We default to high-fidelity restoration prompts
        self.user_prompt = user_prompt or "Enhance the image quality, improve sharpness, reduce noise, and correct lighting while maintaining original content."

    def output_filename(self):
        return "gemini_enhanced.png"

    def process(self):
        """
        Calls Gemini's image editing tool to process the image.
        """
        print(f"[Gemini GenAI] Processing: {self.image_path}")
        
        # 1. Load the original image
        source_img = Image.open(self.image_path)

        try:
            # 2. Call the image_edit tool
            # The model acts as a refiner, taking the source + instructions
            response = genai.image_edit(
                image=source_img,
                prompt=self.user_prompt,
                number_of_images=1
            )

            # 3. Extract the generated image from the response
            # Note: The SDK returns a list of image objects
            enhanced_image = response.images[0]

            # 4. Store for the base class save() method
            self.output_image = enhanced_image
            
            print(f"[Gemini GenAI] Enhancement complete.")
            self.save()

        except Exception as e:
            print(f"Error during Gemini Image Processing: {e}")