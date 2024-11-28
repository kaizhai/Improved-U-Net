from tensorflow import keras
import tensorflow as tf


def block(inputs, filters, rate):
    # 3*3 depth
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same',
                                        dilation_rate=rate, use_bias=False)(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # 1*1 wiseth
    x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def aspp(inputs):
    b, h, w, c = inputs.shape

    x1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    # rate=1
    x2 = block(inputs, filters=512, rate=1)
    # rate=3
    x3 = block(inputs, filters=512, rate=3)
    # rate=5
    x4 = block(inputs, filters=512, rate=5)

    x5 = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    x5 = tf.keras.layers.Reshape(target_shape=[1, 1, -1])(x5)

    x5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x5)
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x5 = tf.keras.layers.Activation('relu')(x5)

    x5 = tf.image.resize(x5, size=(h, w))

    x = tf.keras.layers.concatenate([x1, x2, x3, x4, x5])

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Dropout(rate=0.1)(x)

    return x


## Im-CSAM
def channel_attention(inputs, ratio=0.25):
    channel = inputs.shape[-1]

    x_max = tf.keras.layers.GlobalMaxPooling2D()(inputs)
    x_avg = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    x_max = tf.keras.layers.Reshape([1, 1, -1])(x_max)
    x_avg = tf.keras.layers.Reshape([1, 1, -1])(x_avg)


    x_max = tf.keras.layers.SeparableConv2D(filters=int(channel * ratio), kernel_size=1, activation='relu')(x_max)
    x_avg = tf.keras.layers.SeparableConv2D(filters=int(channel * ratio), kernel_size=1, activation='relu')(x_avg)

    x_max = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=1)(x_max)
    x_avg = tf.keras.layers.SeparableConv2D(filters=channel, kernel_size=1)(x_avg)

    x = tf.keras.layers.Add()([x_max, x_avg])

    x = tf.nn.sigmoid(x)

    x = tf.keras.layers.Multiply()([inputs, x])

    return x



def multi_head_spatial_attention(inputs, num_heads=8):
    _, h, w, c = inputs.shape

    query = tf.keras.layers.Conv2D(filters=c, kernel_size=1)(inputs)
    key = tf.keras.layers.Conv2D(filters=c, kernel_size=1)(inputs)
    value = tf.keras.layers.Conv2D(filters=c, kernel_size=1)(inputs)

    query = tf.reshape(query, [-1, h * w, c])
    key = tf.reshape(key, [-1, h * w, c])
    value = tf.reshape(value, [-1, h * w, c])

    query = tf.reshape(query, [-1, h * w, num_heads, c // num_heads])
    key = tf.reshape(key, [-1, h * w, num_heads, c // num_heads])
    value = tf.reshape(value, [-1, h * w, num_heads, c // num_heads])

    attention = tf.nn.softmax(tf.einsum('bhqd,bhkd->bhqk', query, key) / tf.math.sqrt(float(c // num_heads)), axis=-1)
    context = tf.einsum('bhqk,bhvd->bhqd', attention, value)
    context = tf.reshape(context, [-1, h, w, c])

    x = tf.keras.layers.Add()([inputs, context])
    x = tf.nn.sigmoid(x)
    x = tf.keras.layers.Multiply()([inputs, x])

    return x


def CBAM_attention(inputs):
    x = channel_attention(inputs)
    x = multi_head_spatial_attention(x, num_heads=8)
    return x


def DS_Path(inputs, filters):
    x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(inputs)
    # Depthwise Separable Convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Add the input to the output (residual connection)
    x = tf.keras.layers.Add()([x1, x])
    x = tf.keras.layers.Activation('relu')(x)

    return x


# Dilate-Block
def down_block(x, filters, dropout_rate, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",
                               dilation_rate=1)(x) # dilate1

    aspp_pool = aspp(conv)

    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",
                               dilation_rate=1)(aspp_pool) # dilate2

    pool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=padding)(conv)

    pool = keras.layers.Dropout(dropout_rate)(pool)

    return conv, pool

# traditional,for Ablation experiment
def down_block2(x, filters, dropout_rate, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)
    pool = keras.layers.MaxPool2D((2, 2), (2, 2))(conv)
    pool = keras.layers.Dropout(dropout_rate)(pool)

    return conv, pool


# Expanding block for upsampling
def up_block(x, skip, filters, dropout_rate, kernel_size=(3, 3), padding="same", strides=1):
    us = tf.image.resize(x, size=(tf.shape(x)[1] * 2, tf.shape(x)[2] * 2), method='bicubic')  ## 双三次插值，bilinear为双线性插值
    skip = DS_Path(skip, filters)
    skip = CBAM_attention(skip)
    concat = keras.layers.Concatenate()([us, skip])
    concat = keras.layers.Dropout(dropout_rate)(concat)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    return conv

# Ablation experiment
def up_block2(x, skip, filters, dropout_rate, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    #skip = DS_Path(skip, filters)
    #skip = CBAM_attention(skip)
    concat = keras.layers.Concatenate()([us, skip])
    concat = keras.layers.Dropout(dropout_rate)(concat)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)

    return conv


# Bottlenecking layer
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)
    return conv
