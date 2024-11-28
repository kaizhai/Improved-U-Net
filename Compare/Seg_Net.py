import tensorflow as tf


def encoder_block(inputs, filters, kernel_size=3, pool_size=2):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    pool, pool_indices = tf.nn.max_pool_with_argmax(x, ksize=pool_size, strides=pool_size, padding='SAME')
    return pool, pool_indices

def decoder_block(inputs, filters, kernel_size=3):
    upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(upsample)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def SegNet(image_size):
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    pool1, pool1_indices = encoder_block(inputs, 64)
    pool2, pool2_indices = encoder_block(pool1, 128)
    pool3, pool3_indices = encoder_block(pool2, 256)
    pool4, pool4_indices = encoder_block(pool3, 512)
    pool5, pool5_indices = encoder_block(pool4, 512)

    up5 = decoder_block(pool5, 512)
    up4 = decoder_block(up5, 512)
    up3 = decoder_block(up4, 256)
    up2 = decoder_block(up3, 128)
    up1 = decoder_block(up2, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(up1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

