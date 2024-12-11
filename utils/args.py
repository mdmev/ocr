import argparse

class Arguments:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
            "--model", type=str, default="sonnet",
            help="Model to use for inference. Options: 'sonnet', 'gpt'"
        )
        parser.add_argument(
            "--ground_truth_data", type=str,
            default="json_gt.csv",
            help="Path to the folder containing the ground truth data for the JSON extract data"
        )
        parser.add_argument(
            "--sources_traductor", type=str,
            default="/home/guillfa/CENIA/sources-traductor",
            help="Path to the sources traductor folder."
        )
        parser.add_argument(
            "--output_csv", type=str, default="processed_images.csv",
            help="Path to the output CSV file."
        )
        parser.add_argument(
            "--prompt_type", type=str, default="Classifier",
            help="Prompt type for the model. Options: 'Classifier'."
        )

        return parser.parse_args()
