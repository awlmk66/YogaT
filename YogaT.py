import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import tensorflow as tf

from matplotlib import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import backend as K
from PIL import Image
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
from tensorflow.python.keras.optimizer_v1 import RMSprop
"""
# tensorflow in gpu check
gpu = device_lib.list_local_devices()
print(gpu)

# 데이터 경로 설정
pose_level = []
easy = []
hard = []
pose_level.append(easy)
pose_level.append(hard)
print(pose_level)

train_pose_dir = './yoga_pose_image/training_set'
myPath = os.path.join('./yoga_pose_image/training_set/pigeon')
myPath = os.listdir('./yoga_pose_image/training_set/pigeon')
files = list()

train_pose = os.listdir(train_pose_dir)
myPath = os.listdir(myPath)
print(train_pose[0])
print(myPath[:])

print(os.listdir(train_pose[0] + '/' + myPath[0]))

for i in range(1, len(train_pose)):
    fullPath = os.path.join(myPath, i)
    if os.path.isfile(fullPath):
        files.append(fullPath)
print(files[:5])"""

train_dir = os.path.join('./yoga_pose_image/training_set')
pigeon_dir = os.path.join(train_dir, 'Pigeon')
warrior_dir = os.path.join(train_dir, 'Warrior')
wheel_dir = os.path.join(train_dir, 'Wheel')

"""
# 파일 넣기
myPath = './yoga_pose_image/training_set'

files = list()
for a in os.listdir(myPath):
    fullPath = os.path.join(myPath, a)
    if os.path.isfile(fullPath):
        files.append(fullPath)
print(files[:5])

pigeon_dir2 = os.listdir(pigeon_dir)
p_dir = cv2.imread(pigeon_dir2)
print(p_dir)
"""
# files = os.listdir(pigeon_dir)
# ge = [file for file in files if file.endswith(".*")]
# print("file:{}".format(file_list_jpg))

pigeon_files = os.listdir(pigeon_dir)
warrior_files = os.listdir(warrior_dir)
wheel_files = os.listdir(wheel_dir)

"""pose_level.append(pigeon_files)
pose_level.append(unknown_files)
pose_level.append(warrior_files)
pose_level.append(wheel_files)
print(pose_level[0])"""

print('Total number of training pigeon images:', len(pigeon_files))
print('Total number of training warrior images:', len(warrior_files))
print('Total number of training wheel images:', len(warrior_files))

pic_index = 2

next_pigeon = [os.path.join(pigeon_dir, fname) for fname in pigeon_files[pic_index - 2:pic_index]]
next_warrior = [os.path.join(warrior_dir, fname) for fname in warrior_files[pic_index - 2:pic_index]]
next_wheel = [os.path.join(wheel_dir, fname) for fname in wheel_files[pic_index - 2:pic_index]]

for i, img_path in enumerate(next_pigeon + next_warrior + next_wheel):
    print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()


# generator
# 실제 데이터들은 같은 사이즈로 존재해야함
TRAINING_DIR = "./yoga_pose_image/training_set/"
VALIDATION_DIR = "./yoga_pose_image/test_set/"

IMG_SIZE = 368

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(TRAINING_DIR,
                                              target_size=(368, 368),
                                              class_mode='categorical',
                                              batch_size=5
                                              )
print(len(train_generator))

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(368, 368),
                                                              class_mode='categorical',
                                                              batch_size=5
                                                              )
print(len(validation_generator))

# 데이터 확인
train_path = './yoga_pose_image/training_set'
file_list = os.listdir(train_path)
pose_list = len(file_list)
print(pose_list)

# 모델링
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense((pose_list), activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습세팅
epochs = 300
batch_size = 5
steps_train = len(train_generator)
steps_test = len(validation_generator)
print(steps_train, steps_test)

# 학습 checkpoint로 저장
mc = ModelCheckpoint('./model/YOGAT_Deep_Model(20210615).h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=steps_train,  # 전체 훈련데이터 수 / 배치사이즈
                    validation_data=validation_generator,
                    validation_steps=steps_test,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[es, mc]
                    )
# 모델 save - checkpoint로 저장
# model.save("./model/YOGAT_Deep_Model.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
loss = history.history['loss']
loss_len = np.arange(len(loss))

plt.plot(loss_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(loss_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
plt.plot(loss_len, val_loss, marker='.', c="cornflowerblue", label='Testset_loss')
plt.plot(loss_len, loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()


model = tf.keras.models.load_model('./model/YOGAT_Deep_Model(20210615).h5')

# 예측
categories = os.listdir('./yoga_pose_image/training_set')
print(categories)
nb_classes = len(categories)
print(nb_classes)

# 적용해볼 이미지
pose = input('input name of pose : ')
pose = pose.casefold()
number = input('input number of pose : ')
test_dir = './yoga_pose_image/test_set/' + pose + '/'
test_image = (test_dir + pose + ' (' + number + ').jpg')

# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img_size = (368, 368)
img = img.resize((368, 368))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 368, 368, 3)

# 예측
pred = model.predict(X)
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
print('Predict : ', categories[result[0]])
print(test_image)

