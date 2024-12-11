import os
import csv

def get_list(folder):
    """
    Get a list of images from a folder
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    return [os.path.join(folder, img) for img in os.listdir(folder) 
            if os.path.isfile(os.path.join(folder, img)) and img.lower().endswith(exts)]

def save_to_csv(file_path, data, mode='a'):
    """
    Save incoming data to a CSV file
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode=mode, newline='') as file:
        writer = csv.writer(file)

        if not file_exists or mode == 'w':
            writer.writerow(["image", "candidate", "json", "type"])

        for folder, images in data.items():
            for image_name, info in images.items():
                class_answer = info.get("class_answer", "")
                answer = info.get("answer", "")
                writer.writerow([folder, image_name, class_answer, answer])

def calculate_metrics(results_csv, pairs_candidates_folder="pairs_candidates", sources_traductor_folder="filtered_sources_traductor") -> None:
    """
    Calculate metrics based on the final CSV file
    """

    metrics = {
        pairs_candidates_folder: {"total": 0, "correct": 0},
        sources_traductor_folder: {"total": 0, "correct": 0}
    }

    expected_answers = {
        pairs_candidates_folder: "Candidate",
        sources_traductor_folder: "No Candidate"
    }
    
    with open(results_csv, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            folder = row["Folder"]
            predicted_answer = row["Answer"]
            
            expected_answer = expected_answers.get(folder, None)
            
            if expected_answer is not None:
                metrics[folder]["total"] += 1
                if predicted_answer == expected_answer:
                    metrics[folder]["correct"] += 1

    total_images = sum(folder_metrics["total"] for folder_metrics in metrics.values())
    correct_predictions = sum(folder_metrics["correct"] for folder_metrics in metrics.values())
    overall_accuracy = correct_predictions / total_images if total_images > 0 else 0

    for folder, folder_metrics in metrics.items():
        folder_metrics["accuracy"] = (
            folder_metrics["correct"] / folder_metrics["total"]
            if folder_metrics["total"] > 0
            else 0
        )

    print("Metrics:")
    for folder, folder_metrics in metrics.items():
        print(f"Folder: {folder}")
        print(f"  Total Images: {folder_metrics['total']}")
        print(f"  Correct Predictions: {folder_metrics['correct']}")
        print(f"  Accuracy: {folder_metrics['accuracy'] * 100:.2f}%")
    
    print("\nOverall Metrics:")
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
