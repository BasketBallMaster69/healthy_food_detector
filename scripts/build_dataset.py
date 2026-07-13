from pathlib import Path
import shutil

# Where Food-101 was downloaded
SOURCE = Path(
    "/home/elijahmcgraw/.cache/kagglehub/datasets/"
    "dansbecker/food-101/versions/1/food-101/food-101/images"
)

# Where our new training dataset will go
DESTINATION = Path.home() / "healthy_food_detector" / "dataset"

FOOD_GROUPS = {
    "healthy": [
        "caesar_salad",
        "greek_salad",
        "grilled_salmon",
        "edamame",
        "seaweed_salad",
        "sushi",
    ],
    "unhealthy": [
        "pizza",
        "hamburger",
        "french_fries",
        "donuts",
        "ice_cream",
        "chicken_wings",
    ],
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Food-101 folder not found:\n{SOURCE}")

total_copied = 0

for label, foods in FOOD_GROUPS.items():
    for food in foods:
        source_folder = SOURCE / food
        destination_folder = DESTINATION / label / food

        if not source_folder.exists():
            print(f"Skipped: {food} was not found")
            continue

        destination_folder.mkdir(parents=True, exist_ok=True)

        copied = 0

        for image in source_folder.iterdir():
            if image.is_file() and image.suffix.lower() in VALID_EXTENSIONS:
                shutil.copy2(image, destination_folder / image.name)
                copied += 1

        total_copied += copied
        print(f"{label:9} | {food:20} | {copied} images copied")

print("-" * 50)
print(f"Finished. Total images copied: {total_copied}")
print(f"Dataset location: {DESTINATION}")