import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("marwanabudeeb/rml201610b")

print("Path to dataset files:", path)

# Move dataset to current directory
current_dir = os.getcwd()
dataset_name = "data"
destination = os.path.join(current_dir, dataset_name)

# Copy the entire dataset directory to current path
if os.path.exists(destination):
    print(f"Destination {destination} already exists. Removing it first...")
    shutil.rmtree(destination)

shutil.copytree(path, destination)
print(f"Dataset moved to: {destination}")

# List files in the moved dataset
print("\nFiles in the dataset:")
for root, dirs, files in os.walk(destination):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, destination)
        print(f"  {relative_path}")
