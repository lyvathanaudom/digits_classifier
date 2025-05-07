import os
import pandas as pd
from glob import glob
import shutil
from tqdm import tqdm

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir) # Go up one level from 'scripts' to the project root

source_dir = os.path.join(base_dir, 'data', 'Handwritten-Khmer-Digit')
target_dir = os.path.join(base_dir, 'data', 'datasets', 'samples')
csv_path = os.path.join(target_dir, 'data.csv')

# Create target directory if it doesn't exist
print(f"Creating target directory: {target_dir}")
os.makedirs(target_dir, exist_ok=True)

image_data = []

# Find all image files (assuming .jpg, adjust if other formats exist)
# Use recursive=True to search subdirectories
print(f"Searching for images in: {source_dir}")
image_files = glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)
print(f'Found {len(image_files)} images.')

if not image_files:
    print("No image files found. Please check the source directory path and image format.")
else:
    for img_path in tqdm(image_files, desc='Processing images'):
        try:
            # Extract label from the parent directory name (e.g., '0', '1', ...)
            label = os.path.basename(os.path.dirname(img_path))

            # Get the filename
            filename = os.path.basename(img_path)

            # Define the destination path
            dest_path = os.path.join(target_dir, filename)

            # Copy the image
            # print(f"Copying {img_path} to {dest_path}") # Uncomment for debugging
            shutil.copy2(img_path, dest_path) # copy2 preserves metadata

            # Add data to list
            image_data.append({'filename': filename, 'label': label})
        except Exception as e:
            print(f'Error processing {img_path}: {e}')

    if image_data:
        # Create DataFrame
        print("Creating DataFrame...")
        df = pd.DataFrame(image_data)

        # Save DataFrame to CSV
        print(f"Saving data to CSV: {csv_path}")
        df.to_csv(csv_path, index=False)

        print(f'Successfully processed {len(df)} images.')
        print(f'Data saved to {csv_path}')
        print('\nFirst 5 rows of the CSV:')
        print(df.head())
    else:
        print("No images were successfully processed.")

print("Script finished.")