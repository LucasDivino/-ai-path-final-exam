from .use_cases.get_recommendations import GetRecommendations
from .use_cases.preprocess_user_query import PreprocessUserQuery
from .use_cases.translate_recipes import TranslateRecipes
from .use_cases.generate_recipe_image import GenerateRecipeImage
from .services.llm_client import LLMClient
from .types.recipe import TranslatedRecipes
import asyncio
import argparse
import sys
import logging
from .utils.format_recipes import format_recipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main(query):
        
    if query is None:
        print("Error: No query provided.")
        print("You must provide a food query using the --query parameter.")
        print("Example: python -m src.app --query \"I want a healthy breakfast with eggs\"")
        sys.exit(1)
        
    llm_client = LLMClient()
    preprocessed_query = await PreprocessUserQuery(llm_client=llm_client).preprocess(query)

    _, recipe = GetRecommendations(llm_client=llm_client).get_recommendation(query, preprocessed_query)
    translator = TranslateRecipes()
    image_generator = GenerateRecipeImage()
    
    image_path = await image_generator.generate_image(recipe)

    pt_recipe_promise = translator.translate_to_portuguese(recipe)
    es_recipe_promise = translator.translate_to_spanish(recipe)
    
    pt_recipe_dict, es_recipe_dict = await asyncio.gather(pt_recipe_promise, es_recipe_promise)
    
    translated_recipe: TranslatedRecipes = {
        "original": recipe,
        "pt": pt_recipe_dict,
        "es": es_recipe_dict,
        "image_path": image_path
    }
    
    print(translated_recipe)

    return translated_recipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recipe recommendation system")
    parser.add_argument("--query", type=str, help="Your food request")
    args = parser.parse_args()
    
    result = asyncio.run(main(args.query))
