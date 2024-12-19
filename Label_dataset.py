import pandas as pd
import os
import shutil

# Paths
metadata_path = r"D:\JOURNAL\AUDIO\UrbanSound8K.csv"  
audio_folder = r"D:\JOURNAL\AUDIO"  # Path to the audio folder
output_folder = r"D:\JOURNAL\NEW AUDIO"  # Output folder for organized dataset

# Create output folders for gunshot (1) and non-gunshot (0) sounds
gunshot_folder = os.path.join(output_folder, "1")  # Folder for gunshot sounds
non_gunshot_folder = os.path.join(output_folder, "0")  # Folder for non-gunshot sounds
os.makedirs(gunshot_folder, exist_ok=True)
os.makedirs(non_gunshot_folder, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Initialize a list for labeled data
labeled_data = []

# Iterate through metadata and label files
for index, row in metadata.iterrows():
    file_name = row['slice_file_name']
    fold = row['fold']
    class_id = row['classID']
    
    # Assign label: 1 for gunshot, 0 for others
    label = 1 if class_id == 6 else 0
    labeled_data.append({"file_name": file_name, "label": label})
    
    # Source and destination paths
    src_path = os.path.join(audio_folder, f"fold{fold}", file_name)
    dst_folder = gunshot_folder if label == 1 else non_gunshot_folder
    dst_path = os.path.join(dst_folder, file_name)
    
    # Copy file to the corresponding folder
    shutil.copy2(src_path, dst_path)

# Save the labeled data to a new CSV file
labeled_data_df = pd.DataFrame(labeled_data)
labeled_data_df.to_csv(os.path.join(output_folder, "labeled_data.csv"), index=False)

print("Dataset labeling and organization complete!")