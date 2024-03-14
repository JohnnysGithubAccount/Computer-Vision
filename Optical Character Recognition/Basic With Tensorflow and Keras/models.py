import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Activation, Dense, ReLU, MaxPooling2D, AveragePooling2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from configs import input_shape, dropout_rate, reg


# ResNetBaseModel = ResNet50(weights='imagenet', include_top=False)
# x = ResNetBaseModel.output
# x = tf.keras.layers.GlobalAvgPool2D()(x)
# x = tf.keras.layers.Dense(units=36, activation='softmax')(x)
# ResNet = Model(inputs=ResNetBaseModel.input, outputs=x)

# EfficientNet = EfficientNetB0(weights='imagenet', include_top=False)

def IdentityBlock(x, filters):
    x_skip = x

    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_regularizer=L2(reg))(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_regularizer=L2(reg))(x)
    x = BatchNormalization(axis=3)(x)
    x = Add()([x, x_skip])
    x = ReLU()(x)

    return x

def ConvolutionalBlock(x, filters):
    x_skip = x

    x = Conv2D(filters=filters,
               kernel_size=(3,3),
               strides=(2, 2),
               padding='same',
               kernel_regularizer=L2(reg))(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_regularizer=L2(reg))(x)
    x = BatchNormalization(axis=3)(x)
    x_skip = Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    strides=(2, 2),
                    padding='same',
                    kernel_regularizer=L2(reg),
                    )(x_skip)
    x = Add()([x, x_skip])
    x = ReLU()(x)

    return x

def ResNet34(inputShape, num_classes, block_layers=[3, 4, 6, 3], filters_size=64, units=512):
    x_input = Input(shape=inputShape)

    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(filters=filters_size ,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='same',
               kernel_regularizer=L2(reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    for i in range(len(block_layers)):
        if i==0:
            for j in range(block_layers[i]):
                x = IdentityBlock(x=x, filters=filters_size)
        else:
            filters_size = filters_size * 2
            x = ConvolutionalBlock(x=x, filters=filters_size)
            for j in range(block_layers[i] - 1):
                x = IdentityBlock(x=x, filters=filters_size)

    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    if units != 0:
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=dropout_rate)(x)

    x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=x)

    return model



