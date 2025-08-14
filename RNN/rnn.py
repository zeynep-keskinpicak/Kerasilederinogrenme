# -*- coding: utf-8 -*-


# %% veri setini içeriyeaktar, padding

import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.datasets import imdb  # IMDB veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, roc_curve, auc

import keras_tuner as kt
from keras_tuner.tuners import RandomSearch

import warnings
warnings.filterwarnings("ignore")

# IMDB veri seti yükleniyor - 50000 örnek

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)# num_words=10000 en çok kullanılan 10.000 kelime dahil edilir

# veri on isleme: yorumları aynı uzunluğa getirmek için paddding yontemi kullan

maxlen=100
x_train=pad_sequences(x_train,maxlen=maxlen)  # train verisi uzunluğu ayarla
x_test=pad_sequences(x_test,maxlen=maxlen) # test verisi uzunluğu ayarla



# %% create and compile RNN model

def build_model(hp):
    model=Sequential()
    
    # embeding katmanı: kelimeleri vektörlere çevirir
    model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int("embedding_output",min_value=32,max_value=128,step=32),
                        input_length=maxlen))# vektçr boyutları (32,64,96,128) olabilir
    #simpleRNN: rnn katmani
    model.add(SimpleRNN(units=hp.Int("rnn_units",min_value=32,max_value=128,step=32)))#rnn hücre sayisi 32,64,96,128 olabilir
    
    
    # dDropuot katmani: overfittingi engellemek için rastgele bazı cell'leri kapatır
    model.add(Dropout(rate=hp.Float("dropout_rate",min_value=0.2,max_value=0.5,step=0.1)))
    #cikti katmani: 1 cell ve sigmoid
    model.add(Dense(1,activation="sigmoid")) #♣sigmoid activation fonksiyonu: ikili sınıflandırma  için kullanılır, cikti 0 yada 1 değerini alır
    
    # modelin compile edilmesi
    model.compile(optimizer=hp.Choice("optimizer",["adam","rmsprop"]),
                  loss="binary_crossentropy",# ikili sınıflandırma için kullanılan loss fonksiyonu
                  metrics=["accuracy","AUC"]
                  )
    return model


# %% Hyperparameter search, model train

#hyperparemeter search: random search ile hiperparemetre aranacak

tuner=RandomSearch(
    build_model, #optimize edilecek model foksiyonu
    objective="val_loss", # val_los en düşük olan en iyisidir
    max_trials=2, # farklı model deneyecek
    executions_per_trial=1, #☺her model için1 eğitim denemesi
    directory="rnn_tuner_directory", #modellerin kayıt edileceği dizin
    project_name="imdb_rnn" #projenin adı
    )

#erken durdurma: doğrulama hatası düzelmezse (azlmazsa) eğitimi durdur

early_stopping=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

#modelin eğitimi

tuner.search(x_train, y_train,
             epochs=5,
             validation_split=0.2,#eğitim veri serinin %20 validation olacak
             callbacks=[early_stopping])


# %% evaluate best model

# en iyi modelin alınması

best_model=tuner.get_best_models(num_models=1)[0] # en iyi performans gösteren model

# en iyi modeli kullanarak test et

loss, accuracy, auc_score=best_model.evaluate(x_test,y_test)

print(f"Test Loss: {loss}, test accuracy: {accuracy:.4f}, test auc: {auc_score:.3f}" )

#♥ tahmin yapma ve modelin performansini 

y_pred_prob=best_model.predict(x_test)
y_pred=(y_pred_prob>0.5).astype("int32")#4tahmin edilen değerler 0.5 ten büyükse 1 e küçükse 0 a yuvarlanır

print(classification_report(y_test, y_pred))

#roc eğrisi hesaplama
fpr, tpr,_=roc_curve(y_test, y_pred_prob) #rpc eğrisi için fpr (false positive rate) ve tpr(true positive rate) hesaplanır

roc_auc =auc(fpr,tpr) #roc erisinin altında kalan alan hesaplanır.

#roc eğrisi gorselleştirme

plt.figure()
plt.plot(fpr,tpr,color="darkorange", lw=2, label="ROC Curve (area=%0.2f)" % roc_auc)
plt.plot([0,1],[0,1],color="blue",lw=2,linestyle="--")
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Receveir operating characteristic (Roc) curve")
plt.legend()
plt.show()