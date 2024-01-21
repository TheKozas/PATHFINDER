import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from unet_model import unet_model

# Veri yollarını tanımlayın
data_path = 'C:/Users/ata_k/Desktop/Bitirme/Data Set/archive/train'
sat_images = glob(os.path.join(data_path, '*_sat.jpg'))
mask_images = [f.replace('_sat.jpg', '_mask.png') for f in sat_images]

# Görüntüleri okuyun ve işleyin
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask > 128  # ikili maske oluştur
    return np.expand_dims(mask, axis=-1)

# Veri setlerini oluşturun
X = np.array([load_image(p) for p in sat_images])
y = np.array([load_mask(p) for p in mask_images])

# Eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Modeli oluşturun (unet_model fonksiyonunuzu buraya koyun)
unet = unet_model(input_size=(256, 256, 3))

# Modeli derleyin
unet.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitin
unet.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1)

# Modeli değerlendirin
unet.evaluate(X_test, y_test)
