import aiohttp

class LLMClient:
    def __init__(self):
        self.endpoint = "http://localhost:11434/api/generate"
        self.model = "mistral"

    async def get_response(self, prompt):
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            async with session.post(
                self.endpoint,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["response"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Error calling Ollama API: {response.status} - {error_text}")

    