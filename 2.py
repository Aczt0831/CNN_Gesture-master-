import os
import shutil
import re

# 源路径和目标路径映射
source_path = r'E:\手部识别\CNN_Gesture-master\Gesture_predict'
target_folders = {
    '_0': '666',
    '_4': 'OK',
    '_2': 'punch',
    '_3': 'stop',
    '_1': 'yech'
}

# 确保目标文件夹存在
for folder in target_folders.values():
    target_folder_path = os.path.join(source_path, folder)
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

# 编译正则表达式模式
pattern = re.compile(r'^(.*?)_(\d+)\.jpg$')

# 遍历源路径中的所有文件
for filename in os.listdir(source_path):
    match = pattern.match(filename)
    if match:
        _, suffix = match.groups()
        if f'_{suffix}' in target_folders:
            source_file = os.path.join(source_path, filename)
            target_folder = target_folders[f'_{suffix}']
            target_file = os.path.join(source_path, target_folder, filename)
            shutil.move(source_file, target_file)
            print(f'Moved {filename} to {target_folder}')



