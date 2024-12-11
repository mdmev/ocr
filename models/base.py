from dotenv import load_dotenv
from utils.prompts import prompts, tools

load_dotenv()

class BaseAPI:
    def __init__(self, model_name: str, prompt_type: str, tool: str):
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.prompt_dict = prompts
        self.prompt = None
        self.tool = tools
        self.set_prompt(prompt_type)
        self.set_tool(tool)

    def set_prompt(self, prompt_type: str):
        if prompt_type in self.prompt_dict:
            self.prompt = self.prompt_dict[prompt_type]
        else:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. "
                f"Available types are: {', '.join(self.prompt_dict.keys())}"
            )

    def set_tool(self, tool: str):
        if tool in self.tool:
            self.tool = self.tool[tool]
        else:
            raise ValueError(
                f"Invalid tool: {tool}. "
                f"Available tools are: {', '.join(self.tool.keys())}"
            )


    def run_inference(self, image_data: bytes):
        """
        Implemented in subclasses to send the prompt and image to the respective API.
        Should return a string response.
        """
        raise NotImplementedError
