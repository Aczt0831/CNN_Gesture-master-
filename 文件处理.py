import os
import shutil

path = r'E:\手部识别\CNN_Gesture-master\Gesture_train\\'
Gesturetype = ['666', 'stop', 'yech', 'ok', 'one']

def FileRename(path, Gesture_set):
    num_count = 0
    for ges in Gesture_set:
        file_count = 0
        folder_path = os.path.join(path, ges)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue
        files = os.listdir(folder_path)
        for phone in files:
            file_count += 1
            old_file_path = os.path.join(folder_path, phone)
            new_file_name = f"{num_count}_ges{file_count}.jpg"
            new_file_path = os.path.join(folder_path, new_file_name)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Error renaming {old_file_path}: {e}")
        num_count += 1

def File_to_train_folder(path, Gesture_set, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for ges in Gesture_set:
        folder_path = os.path.join(path, ges)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue
        files = os.listdir(folder_path)
        for phone in files:
            old_file_path = os.path.join(folder_path, phone)
            new_file_path = os.path.join(target_folder, phone)
            try:
                shutil.copy(old_file_path, new_file_path)
                print(f"Copied: {old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Error copying {old_file_path}: {e}")

def lable_rename(path):
    files = os.listdir(path)
    for i in files:
        parts = i.split('_')
        if len(parts) < 2:
            print(f"Skipping invalid filename: {i}")
            continue
        try:
            new_label = str(int(parts[0]) - 1)
            new_file_name = f"{new_label}_{parts[1]}"
            old_file_path = os.path.join(path, i)
            new_file_path = os.path.join(path, new_file_name)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Error renaming {old_file_path}: {e}")
        except ValueError:
            print(f"Invalid number in filename: {i}")

# 示例调用
# FileRename(path, Gesturetype)
# File_to_train_folder(path, Gesturetype, 'E:\\手部识别\\CNN_Gesture-master\\Gesture_train\\')
lable_rename(path)



