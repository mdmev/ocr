import os
import openai

from models.base import BaseAPI

class OpenAIAPI(BaseAPI):
    def __init__(self, model_name: str, prompt_type: str):
        super().__init__(model_name, prompt_type)
        client = openai.OpenAIApi(api_key=os.getenv("OPENAI_KEY"))

    def run_inference(self, image_data: bytes):
        # The OpenAI API's ChatCompletion doesn't directly handle images in messages.
        # If you need image understanding, you'd typically use a vision model or a special endpoint.
        # For the sake of structure, we'll just show prompt usage here.
        
        # If you wanted to combine image data, you'd need a model that accepts image inputs,
        # or provide a description of the image encoded as text.
        
        # For demonstration, just use the prompt as text:
        messages = [
            {"role": "user", "content": self.prompt}
        ]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024
        )

        return response.choices[0].message['content'].strip()
