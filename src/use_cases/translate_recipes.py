from transformers import pipeline
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from src.types.recipe import Recipe

class TranslateRecipes:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pt_translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="eng_Latn",
            tgt_lang="por_Latn",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.es_translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="eng_Latn",
            tgt_lang="spa_Latn",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _translate_text(self, translator, text: str) -> str:
        result = translator(text, max_length=512)
        return result[0]['translation_text']

    def _translate_list(self, translator, items: List[str]) -> List[str]:
        return [self._translate_text(translator, item) for item in items]

    async def _translate_recipe_dict(self, recipe: Recipe, translator) -> Recipe:
        loop = asyncio.get_running_loop()
        
        translated_recipe: Recipe = {
            "title": await loop.run_in_executor(
                self.executor, 
                self._translate_text, 
                translator, 
                recipe['title']
            ),
            "ingredients": await loop.run_in_executor(
                self.executor,
                self._translate_list,
                translator,
                recipe['ingredients']
            ),
            "directions": await loop.run_in_executor(
                self.executor,
                self._translate_list,
                translator,
                recipe['directions']
            ),
            "NER": recipe['NER']
        }
        
        return translated_recipe

    async def translate_to_portuguese(self, recipe: Recipe) -> Recipe:
        return await self._translate_recipe_dict(recipe, self.pt_translator)

    async def translate_to_spanish(self, recipe: Recipe) -> Recipe:
        return await self._translate_recipe_dict(recipe, self.es_translator)
