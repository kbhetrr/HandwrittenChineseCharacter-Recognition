from inception_module import inception
from fire_module import fire_module
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten, Dropout, Dense, Activation, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import Model


def SqueezeNet(input_shape, dropout_rate=0.5):
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=1, name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool1')(x)

    x = fire_module(x, squeeze=16, expand=64, name='fire2')
    x = fire_module(x, squeeze=16, expand=64, name='fire3')

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2')(x)

    x = fire_module(x, squeeze=32, expand=128, name='fire4')
    x = fire_module(x, squeeze=32, expand=128, name='fire5')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3')(x)

    x = fire_module(x, squeeze=48, expand=192, name='fire6')
    x = fire_module(x, squeeze=48, expand=192, name='fire7')
    x = fire_module(x, squeeze=64, expand=256, name='fire8')
    x = fire_module(x, squeeze=64, expand=256, name='fire9')

    x = Dropout(dropout_rate)(x)

    x = AveragePooling2D(pool_size=(7, 7), strides=1, name='avgpool10')(x)
    x = Conv2D(3755, (1, 1), strides=(1, 1), padding='valid', name='conv10', activation='relu')(x)
    x = Flatten()(x)
    x = Activation('softmax', name='softmax')(x)

    return Model(inputs=input_img, outputs=x)

# model
def model():
    input_ = Input(shape=(56, 56, 1))

    x = Conv2D(64, (1, 1), strides=1, padding='same', activation='relu')(input_)
    x = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = inception(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
    x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    #x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    #x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    x = inception(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)

    x = AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    x = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(x)
    x = Flatten()(x)
    #x = Dense(512, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(3755)(x)
    output = Activation('softmax')(x)

    my_model = Model(input_, output)

    return my_model

def model_v2():
    input_ = Input(shape=(56, 56, 1))

    x = Conv2D(64, (1, 1), strides=1, padding='same', activation='relu')(input_)
    x = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = inception(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
    x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    #x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    #x = inception(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    x = inception(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)

    x = Dropout(0.5)(x)
    x = Conv2D(3755, (1, 1), strides=1, padding='valid', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    my_model = Model(input_, output)

    return my_model