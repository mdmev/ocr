import os
import anthropic
import time

from models.base import BaseAPI

class ClaudeAPI(BaseAPI):
    INPUT_COST = 0.000003
    OUTPUT_COST = 0.000015
    
    def __init__(self, model_name: str, prompt_type: str, api_key: str = None, tool: str = None):
        super().__init__(model_name, prompt_type, tool)
        self.api_key = api_key or os.getenv("ANTROPHIC_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in the environment variable ANTHROPIC_KEY.")        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        self.last_message = None
        self.total_cost = 0.0

    def run_inference(self, image, max_retries=3, timeout=15):
        if not hasattr(image, 'get_type') or not hasattr(image, 'get_base64'):
            raise TypeError("Image must have get_type() and get_base64() methods.")


        try:
            start_time = time.time()
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                # tools=self.tool,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.get_type(),
                                    "data": image.get_base64(),
                                },
                            },
                            {
                                "type": "text",
                                "text": self.prompt
                            }
                        ]
                    }
                ]
            )
            duration = time.time() - start_time
            if self.tool:
                return self._clean_output(message.content[0].text)
            if duration > timeout:
                raise TimeoutError(f"Inference exceeded timeout ({timeout}s).")
            self.last_message = message
            return self._clean_output(message.content[0].text)

        except Exception as e:
            if max_retries > 0:
                print(f"Inference failed with error: {e}. Retrying... ({max_retries} retries left)")
                return self.run_inference(image, max_retries=max_retries - 1, timeout=timeout)
            print(f"All inference attempts failed for image -> {image.get_path()}")
            return None

    def calculate_cost(self):
        if self.last_message is None:
            return 0.0
        input_tokens = self.last_message.usage.input_tokens
        output_tokens = self.last_message.usage.output_tokens
        current_cost = (input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST)
        self.total_cost += current_cost
        return current_cost

    def reset_cost(self):
        self.total_cost = 0.0

    def _clean_output(self, raw_output):
        if self.tool:
            lines = raw_output.strip().split("\n")
            # Check for "no candidate" or "candidate"
            for line in lines:
                line_stripped = line.strip().lower()
                if line_stripped == "no candidate":
                    return "No Candidate"
            for line in lines:
                line_stripped = line.strip().lower()
                if line_stripped == "candidate":
                    return "Candidate"

        # Attempt to extract JSON array from raw_output
        start_index = raw_output.find('[')
        end_index = raw_output.rfind(']')

        # If a valid JSON structure is found, return just the JSON portion
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = raw_output[start_index:end_index+1].strip()
            return json_str

        # If no JSON extraction was possible, return the raw output
        return raw_output
