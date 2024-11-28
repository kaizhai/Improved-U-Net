import tensorflow as tf


def FCN_model(image_size, dropout_rate=0.2):
    # Input layer
    input = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # First convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second convolution block
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Output convolutional layer
    output = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model