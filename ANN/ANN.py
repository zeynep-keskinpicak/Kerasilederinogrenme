# -*- coding: utf-8 -*-

# %% veri setinin hazırlanmas ve preprocessing

from keras.datasets import mnist
from keras.utils import to_categorical #kategorik verilere çvirme
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential # sirali model
from keras.layers import Dense #bağlı katmanlar-gizli katmanlar
from keras.models import load_model #modelim yüklenmesi

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#mnist veri setini yukle, eğitim ve tst veri seti olarak ayri ayri yükle

(x_train, y_train), (x_test, y_test)=mnist.load_data()

#♦ilk bir kaç ornegi gorsellestir

plt.figure(figsize=(10,6))

for i in range(6):
    plt.subplot(2, 3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"index: {i}, Label: {y_train[i]}")
    plt.axis("off")
plt.show()

#veri setini normalize edelim, 0-255 arasındaki pixel değerlerini 0-1 arasında olceklendirme

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2])).astype("float32")/255
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2])).astype("float32")/255


# etiketleri kategorik hale çevirme (0-9 arasındaki rakamlara ohe-hot encoding yapiyoruz)
y_train=to_categorical(y_train,10) # 10= sınıf sayısı
y_test=to_categorical(y_test,10)



# %% ANN modelinin olusturlamsı ve derlemesi

model=Sequential()

#ilk katman = 512 tane cell, Relu Activation Function, input size 28*28=784
model.add(Dense(512,activation="relu",input_shape=(28*28,)))

#Wikinci katman:256 cell , activation: tanh
model.add(Dense(256,activation="tanh"))

#output layer: 10 tane olmak zorunda , activation softmax
model.add(Dense(10,activation="softmax"))

model.summary()

# model derlemesi: optimizer (adam:buyuk veri ve kompleks aglar için idealdir)
# model derlemesi: loss (categorical_crossentropy)
# model derlemesi: metrik (accuracy)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# %% Callback'lerin tanimlanmasi ve ANN eğitimleri

# Erken durdurma: eğer val_loss iylesmiyorsa egitimi durduralım
# monitor: doğrulama setindeki (val) kaybi (loss) izler
# patience: 3-> 3 epoch boyunca val loss değişmiyorsa erken durdurma yapalim
# restore_best_weights: en iyi modelin ağırlıklarını geri yükle


early_stopping=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

#model checkpoint: en iyi modelin agirliklarini kaydeder
# save_beest_only en iyi performans gösteren modeli kaydeder

checkpoint=ModelCheckpoint("ann_best_model.keras",monitor="val_loss",save_best_only=True)

#model training: 1 epochs batch size=64 dogrulama orani =%20

history=model.fit(x_train,y_train,
          epochs=10,#veri seti 10 defa eğitilecek
          batch_size=60,# 64 erli ğarçalar ile egitim yapilacak
          validation_split=0.2, # eğitim verisinin 520 si doğrulama verisi olarak kullanılacak
          callbacks=[early_stopping,checkpoint])

# %% Model eveluation, gorsellestirme, model save and load

# test verisi ile model performansı değerlendirme 
# evaluate: modelin test verisi üzerindeki loss(test_loss ve accuracy(test_accuracy) hesaplar

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"test acc: {test_acc}, test loss: {test_loss}")

#training and validation accuracy gorsellestir

plt.figure()
plt.plot(history.history["accuracy"],marker="o",label="Training Accuracy")
plt.plot(history.history["val_accuracy"],marker="o",label="Validation accuracy")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()

#training and validation loss gorsellestirme

plt.figure()
plt.plot(history.history["loss"],marker="o",label="Training Loss")
plt.plot(history.history["val_loss"],marker="o",label="Validation Loss")
plt.title("ANN Loss on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()

#modeli kaydet

model.save("final_mnist_ann_model.keras")

load_model=load_model("final_mnist_ann_model.keras")
