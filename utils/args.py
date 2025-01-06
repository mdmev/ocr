import argparse

class Arguments:
    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
            "--model", type=str, default="sonnet",
            help="Model to use for inference. Options: 'sonnet', 'gpt', 'yolo'"
        )
        parser.add_argument(
            "--dataset", type=str,
            default="datasets/SPA_NHT",
            help="Path to the folder containing the ground truth data for the JSON extract data"
        )
        parser.add_argument(
            "--output", type=str, default="test.json",
            help="Path to the output JSON file."
        )
        parser.add_argument(
            "--prompt_type", type=str, default="nahuatl",
            help="Prompt type for the model."
        )

        return parser.parse_args()
