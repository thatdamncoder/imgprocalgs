from imgprocalgs.algorithms.base import BaseImageAlgorithm
from PIL import Image
from google import genai
from google.genai import types
import os

class GeminiGhibliConverter(BaseImageAlgorithm):
    """
    Convert image to Ghibli-style using Gemini (Imagen)
    """

    def __init__(self, image_path: str, destination_path: str, user_prompt: str = None):
        super().__init__(image_path, destination_path)
        self.user_prompt = user_prompt or "In the style of Studio Ghibli, hand-drawn anime, lush landscapes, soft pastel colors, whimsical atmosphere"
        # Initialize the Gemini Client
        # Ensure GEMINI_API_KEY is set in your environment variables
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def output_filename(self):
        return "gemini_ghibli.png"

    def process(self):
        print(f"[Gemini] Converting to Ghibli style with prompt: {self.user_prompt}")

        # 1. Load the reference image
        raw_image = Image.open(self.image_path)

        try:
            # 2. Call the Image-to-Image (Image Editing) functionality
            # This uses the image as a structural guide to apply the new style
            response = self.client.models.edit_image(
                model='imagen-3.0-capability-001', # Or latest available Imagen model
                image=raw_image,
                prompt=self.user_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    include_rai_reason=True,
                    output_mime_type='image/png'
                )
            )

            # 3. Extract the generated image
            # The SDK returns a generated_image object containing the PIL image
            output_image = response.generated_images[0].image

            # 4. Save using the base class method
            self.save(output_image)
            print("[Gemini] Successfully saved Ghibli-style image.")

        except Exception as e:
            print(f"Error during Gemini processing: {e}")