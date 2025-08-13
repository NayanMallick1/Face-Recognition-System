import os

def rename_files(directory, person_name):
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    index = 1
    for file in files:
        # Extract the file extension
        file_extension = os.path.splitext(file)[1].lower()
        
        # Only process .jpg and .png files
        if file_extension in ['.jpg', '.png']:
            # Create the new file name
            new_file_name = f"{person_name}{index}.jpeg"
            
            # Get full paths
            old_file_path = os.path.join(directory, file)
            new_file_path = os.path.join(directory, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            
            print(f"Renamed: {file} -> {new_file_name}")
            index += 1

# Usage example
directory_path = "faces_dataset/adrija_sengupta"  # Replace with your directory path
person_name = "as"                 # Replace with the desired prefix
rename_files(directory_path, person_name)
