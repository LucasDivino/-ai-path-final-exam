from src.services.llm_client import LLMClient
from src.types.recipe import PreprocessedQuery
from src.utils.prompts import preprocess_user_query
import json

class PreprocessUserQuery:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def preprocess(self, query: str) -> PreprocessedQuery:
        prompt = preprocess_user_query(query)
        response = await self.llm_client.get_response(prompt)
        
        try:
            parsed_response = json.loads(response) if isinstance(response, str) else response
            
            required_fields = ["simplified_request", "keywords", "restrictions"]

            if not all(field in parsed_response for field in required_fields):
                missing_fields = [field for field in required_fields if field not in parsed_response]
                raise ValueError(f"Missing required fields in response: {', '.join(missing_fields)}")
              
            return parsed_response
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from OpenAI")
        except Exception as e:
            raise ValueError(f"Error processing response: {str(e)}")
