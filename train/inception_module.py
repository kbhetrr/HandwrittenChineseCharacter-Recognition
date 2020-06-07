from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, ZeroPadding2D

# inception module
def inception(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    """
    # Arguments
    x : 입력이미지
    o_1 : 1x1 convolution 연산 출력값의 채널 수
    r_3 : 3x3 convolution 이전에 있는 1x1 convolution의 출력값 채널 수
    o_3 : 3x3 convolution 연산 출력값의 채널 수
    r_5 : 5x5 convolution 이전에 있는 1x1 convolution의 출력값 채널 수
    o_5 : 5x5 convolution 연산 출력값의 채널 수
    pool: maxpooling 다음의 1x1 convolution의 출력값 채널 수

    # returns
    4 종류의 연산의 결과 값을 채널 방향으로 합친 결과
    """
    conv_1x1 = Conv2D(o_1, (1, 1), strides=1, padding='same', activation='relu')(x)

    conv_3x3_reduce = Conv2D(r_3, (1, 1), strides=1, padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(o_3, (3, 3), strides=1, padding='same', activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(r_5, (1, 1), strides=1, padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(o_5, (5, 5), strides=1, padding='same', activation='relu')(conv_5x5_reduce)

    maxpool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    maxpool_proj = Conv2D(pool, (1, 1), strides=1, padding='same', activation='relu')(maxpool)

    return concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj])
