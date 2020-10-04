# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:52:54 2020

@author: Gaurav
"""

from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import Xception
train_datagen=ImageDataGenerator(brightness_range=[0.1,0.4],
                           zoom_range=[0.2,0.5],
                           horizontal_flip=True,
                           rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train=train_datagen.flow_from_directory(
    "E:/All Data Set/Mask Dedection/dest_folder/train",
    target_size=(224,224),
    batch_size=10,
    class_mode="categorical",
    # color_mode="grayscale"
    )

test=test_datagen.flow_from_directory(
    "E:/All Data Set/Mask Dedection/dest_folder/test",
    target_size=(224,224),
    batch_size=10,
    class_mode="categorical",
    # color_mode="grayscale"
    )

baseModel=Xception(include_top=False,input_shape=[224,224,3])

# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

history=model.fit(train,validation_data=test,epochs=10,steps_per_epoch=len(train),validation_steps=len(test))
print(history.history.keys())
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()

model.save("E:/AI Application Implementation/Face_mask_detection/mask_detection.model")



