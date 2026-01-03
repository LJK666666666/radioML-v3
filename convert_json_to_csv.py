import json
import csv

# Read the JSON file
json_file_path = r'd:\010_CodePrograms\R\radioML-v4\github\radioML-v4\output\models\logs\cgdnn_model_stratified_detailed_log.json'
csv_file_path = r'd:\010_CodePrograms\R\radioML-v4\github\radioML-v4\output\models\logs\cgdnn_model_stratified_detailed_log.csv'

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract epochs data
epochs = data['epochs']

# Define CSV headers
headers = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 
           'epoch_time_seconds', 'learning_rate', 'timestamp']

# Write to CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    
    # Write header
    writer.writeheader()
    
    # Write each epoch's data
    for epoch_data in epochs:
        writer.writerow(epoch_data)

print(f"Successfully converted JSON to CSV!")
print(f"Total epochs: {len(epochs)}")
print(f"CSV file saved to: {csv_file_path}")
