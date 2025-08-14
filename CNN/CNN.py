# -*- coding: utf-8 -*-


# %% veri setini içeriye ve preprocessing : normalizasyon, one-hot encoding

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10 #veri seti
from tensorflow.keras.utils import to_categorical #one-hot encoding
from tensorflow.keras.models import Sequential # sıralı model
from tensorflow.keras.layers import Conv2D, MaxPooling2D # feature extraction 
from tensorflow.keras.layers import Flatten, Dense, Dropout #clasification
from tensorflow.keras.optimizers import  RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
                                            

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#gorsellestirme
class_labels=["airplane","automobile","Bird","cat","Deer","Dog","Frog","Horse","Ship","Truck"]

fig, axes=plt.subplots(1, 5,figsize=(15,10))

for i in range(5):
    axes[i].imshow(x_train[i])
    label=class_labels[int(y_train[i])]
    axes[i].axis("off")
plt.show()

# veri seti normalizasyon

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

#♣one-hot encoding

y_train=to_categorical(y_train,10) #10 class var, bu nedenle 10 yazıyoruz
y_test=to_categorical(y_test,10)



#%% Veri attirma (Data Augmentation )

datagen=ImageDataGenerator(
    rotation_range=20, #20 dereceye kadar döndürme sağlar
    width_shift_range=0.2, #görüntüyü yatayda %20 kaydırma
    height_shift_range=0.2, # görüntüyü dikeyde %20 kaydırma
    shear_range=0.2, #görüntü üzerinde kaydırma
    zoom_range=0.1, #görüntüye zoom uyguglama
    horizontal_flip=True, #görünyü yatayda ters çevirme simetriği
    fill_mode="nearest" #boş alanları doldurmak için en yakın pixel değerlerini kullan
    )

datagen.fit(x_train) #data augmentation eğitim verileri üzerinde uygula



#%% Create, compile, and train model

#cnn model olustur( base model)

model=Sequential()

# Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT

model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # bağlantıların %25'ine rasgele olarak kapat (overfitting engelleme)

# Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(62,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Classification : FLATTEN DENSE, RELU, DROPOUT, DENSE (OUTPUY LAYER)

model.add(Flatten())# vektör oluştur
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

model.summary()

# model derleme (compile)

model.compile(optimizer=RMSprop(learning_rate=0.0001,decay=1e-6),# learning_rate: öğrenme hızı ,decay:öğrenme hızını her seferinde küçültecek 
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#model training

history=model.fit(datagen.flow(x_train,y_train,batch_size=250),
          epochs=10, #eğitim dönen sayisi
          validation_data=(x_test,y_test) #doğrulama seti
          ) 

#%% Test model and evaluate performance

#modelin test veri seti üzerinden tahminini yap

y_pred=model.predict(x_test) #y_pred=[0,1] mesela 0.8 cikarsa %0 olma olasılığı olarak tariflendirir
np.argmax(y_pred,axis=1) #◘tahmin edilen sınıfları almak için

y_pred=model.predict(x_test) # y_pred=[0,1] mesela 0.8 olma olasılığı %80 olarak tarif edilir
y_pred_class=np.argmax(y_pred,axis=1) # tahmin edilen sınıfları al
y_true=np.argmax(y_test,axis=1)

#clasification report hesapla

report=classification_report(y_true, y_pred_class,target_names=class_labels)
print(report)

plt.figure(figsize=(10,8))
#kayip grafikleri
plt.subplot(1,2,1) #1 satır 2 sütün 1. subplot
plt.plot(history.history["loss"],label="Train loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation loss")
plt.legend()
plt.grid()

#accuracy
plt.subplot(1,2,2) #1 satır 2 sütün 1. subplot
plt.plot(history.history["accuracy"],label="Train accuacy")
plt.plot(history.history["val_accuracy"],label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
