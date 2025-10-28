import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from PIL import Image
import os
import random
from keras import backend as K
from keras import regularizers
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model

from sklearn.metrics import confusion_matrix
import itertools


class GestureRecognition:

    def __init__(self, batch_size, epochs, num_classes, train_folder, test_folder, model_name, class_types):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.model_name = model_name
        self.class_types = class_types
        self.shape1 = 100
        self.shape2 = 100

    def read_images(self, folder):
        image_list = []
        label_list = []
        valid_extensions = ('.jpg', '.jpeg', '.png')

        for label_string in os.listdir(folder):
            label_path = os.path.join(folder, label_string)
            if not os.path.isdir(label_path):
                print(f"Skipping non-directory: {label_path}")
                continue

            try:
                label_index = self.class_types.index(label_string)

                for file in os.listdir(label_path):
                    if not file.lower().endswith(valid_extensions):
                        print(f"Skipping non-image file: {os.path.join(label_path, file)}")
                        continue

                    image_path = os.path.join(label_path, file)
                    image = Image.open(image_path).convert('L')
                    image = np.array(image).reshape(self.shape1, self.shape2, 1)
                    image_list.append(image)
                    label_list.append(label_index)

            except ValueError:
                print(f"Label '{label_string}' not found in class types list.")
            except Exception as e:
                print(f"Error processing directory {label_path}: {e}")

        return image_list, label_list

    def train(self):
        train_image_list, train_label_list = self.read_images(folder=self.train_folder)
        test_image_list, test_label_list = self.read_images(folder=self.test_folder)

        test_image_list, test_label_list = np.array(test_image_list).astype('float32') / 255, np.array(test_label_list)

        indices = [i for i in range(len(train_image_list))]
        random.shuffle(indices)
        for i in range(len(train_image_list)):
            j = indices[i]
            train_image_list[i], train_image_list[j] = train_image_list[j], train_image_list[i]
            train_label_list[i], train_label_list[j] = train_label_list[j], train_label_list[i]

        train_image_list = np.array(train_image_list).astype('float32') / 255
        train_label_list = np.array(train_label_list)

        train_label_list = to_categorical(train_label_list, self.num_classes)
        test_label_list = to_categorical(test_label_list, self.num_classes)

        model = Sequential()
        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            padding='valid',
            input_shape=(self.shape1, self.shape2, 1),
            name='conv_layer_1'
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_1'))
        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            name='conv_layer_2'
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_2'))

        model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='max_pooling_layer_1'
        ))
        model.add(Dropout(0.5, name='dropout_1'))
        model.add(Flatten(name='flatten_1'))
        model.add(Dense(128, name='dense_layer_1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_3'))
        model.add(Dropout(0.5, name='dropout_2'))

        model.add(Dense(self.num_classes,
                       kernel_regularizer=regularizers.l2(0.01),
                       name='dense_layer_2'))
        model.add(Activation('softmax', name='activation_4'))

        adam = Adam(learning_rate=0.001)  # Modify here

        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.summary()
        model.get_config()
        plot_model(model, to_file='model.png', show_shapes=True)
        history = model.fit(
            x=train_image_list,
            y=train_label_list,
            epochs=self.epochs,
            validation_split=0.33,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(test_image_list, test_label_list)
        )

        predictions = model.predict(test_image_list)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(test_label_list, axis=1)

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        self.plot_confusion_matrix(conf_matrix, class_range=range(self.num_classes))

        self.visualize_history(history)
        model.save(self.model_name)
        K.clear_session()

    def plot_confusion_matrix(self, cm, class_range, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure(1, figsize=(7, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        gesture_types = self.class_types

        tick_marks = np.arange(len(class_range))
        plt.xticks(tick_marks, gesture_types, rotation=45)
        plt.yticks(tick_marks, gesture_types)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def visualize_history(self, hist):
        train_loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        train_acc = hist.history['accuracy']
        val_acc = hist.history['val_accuracy']
        epochs_range = range(self.epochs)

        plt.figure(2, figsize=(7, 5))
        plt.plot(epochs_range, train_loss)
        plt.plot(epochs_range, val_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Validation Loss')
        plt.grid(True)
        plt.legend(['Train', 'Validation'])

        plt.figure(3, figsize=(7, 5))
        plt.plot(epochs_range, train_acc)
        plt.plot(epochs_range, val_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy vs Validation Accuracy')
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc=4)

        plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gesture_types = ['666', 'OK', 'punch', 'stop', 'yech']
    training_instance = GestureRecognition(batch_size=32, epochs=20, num_classes=5,
                                          train_folder=r'E:\手部识别\CNN_Gesture-master\Gesture_train\\',
                                          test_folder=r'E:\手部识别\CNN_Gesture-master\Gesture_predict\\',
                                          model_name=r'E:\手部识别\CNN_Gesture-master\Gesture_1.h5',
                                          class_types=gesture_types)
    training_instance.train()
    K.clear_session()



