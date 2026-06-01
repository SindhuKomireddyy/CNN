import os
from PIL import Image

dataset_path = r"C:\Users\sindh\Downloads\archive (20)\PetImages"

removed_count = 0

for folder in ["Cat", "Dog"]:

    folder_path = os.path.join(dataset_path, folder)

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        try:

            img = Image.open(file_path)

            img = img.convert("RGB")

            img.save(file_path)

        except:

            try:
                os.remove(file_path)
                removed_count += 1

            except:
                pass

print(f"Removed {removed_count} corrupted images")
print("Dataset cleaned successfully")