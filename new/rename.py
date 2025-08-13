import os

# ✅ FIXED PATH
folder_path = r"new\Penguin"

prefix = "as"
images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
images.sort()

for index, image in enumerate(images, start=1):
    old_path = os.path.join(folder_path, image)
    new_name = f"{prefix}{index}.jpeg"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {image} → {new_name}")

print("\n✅ All images renamed successfully!")
