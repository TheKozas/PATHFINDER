import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # İndirgeme yolu
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    # Köprü
    c5 = conv_block(p4, 1024)

    # Genişleme yolu
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)
    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)
    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Modeli oluştur
unet = unet_model()
unet.summary()
