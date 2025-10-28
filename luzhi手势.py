#按下‘l’录制手势
#按下‘b’重置背景减除器
#按下‘t'开始训练模型
#按下‘p'开始预测手势
#按下’exit‘返回主循环
#按下‘q'退出程序11
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import os
from tensorflow.keras import backend
import time
import random

class Training:
    def __init__(self, batch_size, epochs, categories, train_folder, test_folder, model_name, type):
        self.batch_size = batch_size
        self.epochs = epochs
        self.categories = categories
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.model_name = model_name
        self.type = type

    def create_model(self):
        inputs = Input(shape=(100, 100, 1))
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.categories, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_folder,
            target_size=(100, 100),
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = test_datagen.flow_from_directory(
            self.test_folder,
            target_size=(100, 100),
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        model = self.create_model()
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            epochs=self.epochs
        )
        model.save(self.model_name, save_format='tf')  # 使用TensorFlow格式保存模型

class Gesture():
    def __init__(self, train_path, predict_path, gesture, train_model):
        self.blurValue = 5
        self.bgSubThreshold = 36
        self.train_path = train_path
        self.predict_path = predict_path
        self.threshold = 60
        self.gesture = gesture
        self.train_model = train_model
        self.skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.x1 = 380
        self.y1 = 60
        self.x2 = 640
        self.y2 = 350

    def collect_gesture(self, capture, ges, photo_num):
        global p_model  # 声明 p_model 为全局变量
        p_model = None  # 初始化 p_model 为 None
        photo_num = photo_num
        video = False
        predict = False
        count = 0
        cap = cv2.VideoCapture(capture)
        cap.set(10, 200)
        bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.imshow('Original', frame)
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            rec = cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2)
            frame = frame[self.y1:self.y2, self.x1:self.x2]
            cv2.imshow('bilateralFilter', frame)
            bg = bgModel.apply(frame, learningRate=0)
            cv2.imshow('bg', bg)
            fgmask = cv2.erode(bg, self.skinkernel, iterations=1)
            cv2.imshow('erode', fgmask)
            bitwise_and = cv2.bitwise_and(frame, frame, mask=fgmask)
            cv2.imshow('bitwise_and', bitwise_and)
            gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 2)
            cv2.imshow('GaussianBlur', blur)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow('thresh', thresh)
            Ges = cv2.resize(thresh, (100, 100))

            if predict and p_model is not None:
                img = np.array(Ges).reshape(-1, 100, 100, 1) / 255
                prediction = p_model.predict(img)
                final_prediction = [result.argmax() for result in prediction][0]
                ges_type = self.gesture[final_prediction]
                cv2.putText(rec, ges_type, (self.x1, self.y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2, color=(0, 0, 255))

            cv2.imshow('Original', rec)
            if video and count < photo_num:
                filename = '{}_{}.jpg'.format(str(random.randrange(1000, 100000)), str(ges))
                full_path = os.path.join(self.train_path, self.gesture[ges], filename)
                cv2.imencode('.jpg', Ges)[1].tofile(full_path)
                count += 1
                print(count)
            elif count == photo_num:
                print('{}张测试集手势录制完毕，3秒后录制此手势测试集，共{}张'.format(photo_num, int(photo_num * 0.43)))
                time.sleep(3)
                count += 1
            elif video and photo_num < count < int(photo_num * 1.43):
                filename = '{}_{}.jpg'.format(str(random.randrange(1000, 100000)), str(ges))
                full_path = os.path.join(self.predict_path, self.gesture[ges], filename)
                cv2.imencode('.jpg', Ges)[1].tofile(full_path)
                count += 1
                print(count)
            elif video and count >= int(photo_num * 1.43):
                video = False
                ges += 1
                if ges < len(self.gesture):
                    print('此手势录制完成，按l录制下一个手势')
                else:
                    print('手势录制结束, 按t进行训练')

            k = cv2.waitKey(10)
            if k == 27:  # 按ESC键退出
                break
            elif k == ord('l'):  # 录制手势
                video = True
                count = 0
            elif k == ord('p'):  # 预测手势
                predict = True
                while True:
                    model_name = input('请输入模型的名字:\n')
                    if model_name == 'exit':
                        break
                    if model_name.endswith('.h5') or model_name.endswith('.keras'):
                        if model_name in os.listdir('./'):
                            print('正在加载{}模型'.format(model_name))
                            p_model = load_model(model_name)
                            break
                        else:
                            print('模型名字输入错误，请重新输入，或输入exit退出')
                    else:
                        print('模型文件名应以 .h5 或 .keras 结尾')
            elif k == ord('b'):  # 重置背景减除器
                bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
                print('背景重置完成')
            elif k == ord('t'):  # 开始训练模型
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                train = Training(batch_size=32, epochs=5, categories=len(self.gesture), train_folder=self.train_path,
                                 test_folder=self.predict_path, model_name=self.train_model, type=self.gesture)
                train.train()
                backend.clear_session()
                print(f'{self.train_model}模型训练结束')

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']
    train_path = '2/'
    predict_path = '3/'

    for path in [train_path, predict_path]:
        if not os.path.exists(path):
            os.mkdir(path)
        for gesture in Gesturetype:
            if not os.path.exists(os.path.join(path, gesture)):
                os.mkdir(os.path.join(path, gesture))

    print(f'训练手势有：{Gesturetype}')
    train_model = 'Gesture2.h5'
    Ges = Gesture(train_path, predict_path, Gesturetype, train_model)
    num = 5
    x = 0
    Ges.collect_gesture(capture=0, ges=x, photo_num=num)



