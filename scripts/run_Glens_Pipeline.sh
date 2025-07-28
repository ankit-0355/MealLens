#!/bin/bash

# Optional: activate your Python environment
# source /path/to/your/venv/bin/activate
PYTHON=$(which python)

echo "Running Segmentation Pipeline..."
$PYTHON C:\\Users\\sahil\\Desktop\\Meal_final\\scripts\\run_gui.py

echo "Running Google Lens Scrapper..."
$PYTHON C:\\Users\\sahil\\Desktop\\Meal_final\\scripts\\Google_lens_Scrapper.py

echo "Running LLM Analysis Script..."
$PYTHON C:\\Users\\sahil\\Desktop\\Meal_final\\scripts\\LLM_Analysis_Script.py

echo "Running Labelling Pipeline..."
$PYTHON C:\\Users\\sahil\\Desktop\\Meal_final\\scripts\\annotate_labels_paths.py

echo "Pipeline completed."

# Path to the image you want to open
IMAGE_PATH="C:\\Users\\sahil\\Desktop\\Meal_final\\output\\fridge_image__2_\\output_labeled.jpg"  # update this path if your output file has a different name

# Platform-specific way to open an image
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$IMAGE_PATH"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open "$IMAGE_PATH"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    start "" "$IMAGE_PATH"
else
    echo "Unsupported OS. Please open the image manually at: $IMAGE_PATH"
fi

echo "Pipeline completed."