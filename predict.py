from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# 加载预训练模型
model_path = r'E:\手部识别\CNN_Gesture-master\Gesture_2.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)
Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']

# 重新编译模型以更新度量指标（如果需要）
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义主路径
main_path = r'E:\手部识别\CNN_Gesture-master\Gesture_predict\\'

file_count = 0

for gesture_folder in Gesturetype:
    folder_path = os.path.join(main_path, gesture_folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping non-directory: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        # 检查文件扩展名是否为 .jpg 或 .jpeg 或 .png
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {os.path.join(folder_path, file)}")
            continue

        try:
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('L')  # 转换为灰度图像
            img = img.resize((100, 100))  # 调整图像大小到 100x100

            # 将图像转换为数组并归一化
            img_array = np.array(img).reshape(-1, 100, 100, 1) / 255.0

            # 进行预测
            prediction = model.predict(img_array)
            final_prediction = np.argmax(prediction)

            # 提取实际标签
            actual_label = Gesturetype.index(gesture_folder)

            # 比较预测结果与实际标签
            if final_prediction != actual_label:
                print(f'文件 {file} 的预测结果: {Gesturetype[final_prediction]}')
                probabilities = [f'{Gesturetype[i]}的概率: {prediction[0][i] * 100:.2f}%' for i in range(len(Gesturetype))]
                print(probabilities)
                file_count += 1

        except Exception as e:
            print(f"Error processing file {os.path.join(folder_path, file)}: {e}")

print(f'Total mismatched predictions: {file_count}')



