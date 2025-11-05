# predicts exactly num_chars characters, one-hot per character

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x

    # First conv
    x = Conv2D(
        filters, kernel_size, strides=stride, padding="same", use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second conv
    x = Conv2D(
        filters, kernel_size, strides=1, padding="same", use_bias=False
    )(x)
    x = BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(
            filters, (1, 1), strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add skip connection
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


def build_cnn_backbone(input_shape):
    inputs = Input(shape=input_shape)

    # Initial conv
    x = Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Residual blocks with downsampling
    x = residual_block(x, 32, stride=1)
    x = residual_block(x, 32, stride=2)  # downsample

    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=2)

    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=2)

    # Flatten and dense layer
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    return Model(inputs=inputs, outputs=x, name="residual_cnn_backbone")


def add_output_heads(backbone, num_chars, num_classes):
    heads = []
    for i in range(num_chars):
        head = Dense(num_classes, activation="softmax", name=f"char_{i+1}")(
            backbone.output
        )
        heads.append(head)

    return Model(inputs=backbone.input, outputs=heads, name="multi_head_model")


def build_model(input_shape, num_chars=5, num_classes=19):
    backbone = build_cnn_backbone(input_shape)
    model = add_output_heads(backbone, num_chars, num_classes)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=["categorical_crossentropy"] * num_chars,
        metrics=["accuracy"] * num_chars,
    )

    return model
