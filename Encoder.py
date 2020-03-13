from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D                                                                                                                                                         
from keras.models import Model
from keras.models import load_model
#from keras.callbacks import TensorBoard
#from keras.datasets import mnist
import numpy as np
from keras.callbacks import ModelCheckpoint

Train = False

def train_model():
    input_img = Input(shape=(128, 128, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    #Model
    autoencoder = Model(input_img, decoded)

    #optimizer, loss
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    #Train
    history = autoencoder.fit(x_train, x_train,
                    epochs=5,
                    batch_size=128,
                    shuffle=True,
                    callbacks=[ModelCheckpoint('./dlc/data_model/{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)])


    autoencoder.save('./dlc/autoencoder.h5')
    print(history)


x_train = np.load('./dlc/data.npy')

if Train:
    train_model()
else:
    model = load_model('./dlc/autoencoder.h5')
    encoder_model = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    feature = encoder_model.predict(x_train)
    print(feature.shape)

np.save('./dlc/encoder_feature.npy', feature)
