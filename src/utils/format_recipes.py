from typing import Dict, List, Union

def format_recipe(recipe: Dict[str, Union[str, List[str]]]) -> str:
    parts = [f"title: {recipe['title']}"]
    
    if recipe.get('ingredients'):
        parts.append("ingredients:")
        parts.extend([f"- {ingredient}" for ingredient in recipe['ingredients']])
    
    if recipe.get('directions'):
        parts.append("directions:")
        parts.extend([f"- {direction}" for direction in recipe['directions']])
    
    return "\n".join(parts)
