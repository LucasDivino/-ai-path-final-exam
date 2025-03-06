from typing import TypedDict, List, Optional

class Recipe(TypedDict, total=False):
    title: str
    ingredients: List[str]
    directions: List[str]

class TranslatedRecipes(TypedDict):
    original: Recipe
    pt: Recipe
    es: Recipe 
    image_path: Optional[str]

class PreprocessedQuery(TypedDict):
    simplified_request: str
    keywords: List[str]
    restrictions: List[str]