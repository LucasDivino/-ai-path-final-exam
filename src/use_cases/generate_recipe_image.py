from diffusers import StableDiffusionPipeline
import torch
from src.types.recipe import Recipe
from src.utils.prompts import create_recipe_image_prompt
import logging
from PIL import Image
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class GenerateRecipeImage:
    def __init__(self):
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        self.images_dir = "static"
        os.makedirs(self.images_dir, exist_ok=True)

    def _save_image(self, image: Image.Image, recipe: Recipe) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(x for x in recipe['title'] if x.isalnum() or x in [' ', '-', '_']).rstrip()
        filename = f"{safe_title}_{timestamp}.png"
        filepath = os.path.join(self.images_dir, filename)
        
        image.save(filepath)
        return filepath

    async def generate_image(self, recipe: Recipe) -> str:
        try:
            prompt = create_recipe_image_prompt(recipe['title'], recipe['NER'])
            
            with torch.inference_mode():
                image = self.pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
            
            image_path = self._save_image(image, recipe)    
            return image_path
            
        except Exception as e:
            logger.error(f"Error generating image for recipe '{recipe['title']}': {str(e)}")
            return recipe
