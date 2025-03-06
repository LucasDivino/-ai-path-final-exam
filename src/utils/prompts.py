from typing import List


def preprocess_user_query(query: str) -> str:
    return f"""You are a assistant that helps to process user recipes requests. Your task is to:
    1. Simplify the user request, making it more concise and remove any information that is not relevant to the recipe search
    2. Extract the keywords of the request (keywords must be single words)
    3. Extract any restrictions that the request might have (restrictions must be in positive form)

    Example:
    - If user says "I want to cook a healthy meal for my family" -> simplify to "healthy meal"
    - If user says "no gluten" -> add "gluten" to restrictions list

    You should return ONLY a JSON object with this format:
    {{
        "simplified_request": "string",
        "keywords": ["string"],
        "restrictions": ["string"]
    }}

    User request: {query}
    """

def create_recipe_image_prompt(recipe_title: str, recipe_ingredients: List[str]) -> str:
    ingredients_text = ", ".join(recipe_ingredients[:5])
    if len(recipe_ingredients) > 5:
        ingredients_text
    
    return f"""You are a food photography expert. Create an image with these characteristics:
    1. Subject: Homemade {recipe_title} with {ingredients_text}
    2. Lighting: Warm and natural
    3. Setting: Rustic kitchen, cozy atmosphere
    4. Style: Comfort food, homestyle cooking
    5. Presentation: Natural on a simple plate
    6. Quality: Realistic, appetizing appearance
    """

def build_recipe_selection_prompt(original_query, preprocessed_query, candidate_recipes):
    prompt = f"""You are a culinary expert helping to select the perfect recipe. Consider this request:

    User Query: "{original_query}"

    Request Analysis:
    1. Simplified Query: "{preprocessed_query['simplified_request']}"
    2. Keywords: {', '.join(preprocessed_query['keywords']) if preprocessed_query['keywords'] else 'None'}
    3. Restrictions: {', '.join(preprocessed_query.get('restrictions', [])) if preprocessed_query.get('restrictions', []) else 'None'}

    Available Recipes:
    """
    
    for i, recipe in enumerate(candidate_recipes, 1):
        payload = recipe["payload"]
        recipe_name = payload.get("name", "No name")
        recipe_ingredients = payload.get("ingredients", [])
        recipe_instructions = payload.get("instructions", "")
        
        has_restrictions = recipe.get("has_restrictions", False)
        matched_restrictions = recipe.get("matched_restrictions", [])
        
        prompt += f"""Recipe {i} (ID: {recipe["id"]}):
    1. Name: {recipe_name}
    2. Ingredients: {', '.join(recipe_ingredients)}
    3. Instructions: {recipe_instructions[:150]}{"..." if len(recipe_instructions) > 150 else ""}
    4. Dietary Notes: {"Contains: " + ', '.join(matched_restrictions) if has_restrictions else "No restrictions"}

    """
    
    prompt += """You should return ONLY the ID of the best matching recipe.
    If no perfect match exists, choose the closest option.
    
    Response format: "ID: [recipe id]"
    """
    
    return prompt