import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
)
from pathlib import Path
import joblib, tensorflow as tf

def residual_block(x, filters, kernel_size=(5, 5),
                   downsample: bool = False,
                   dropout_rate: float = 0.0):
    """A two‑conv residual block with optional down‑sampling."""
    shortcut = x

    # 1st conv
    x = Conv2D(filters, kernel_size,
               strides=(2, 2) if downsample else (1, 1),
               padding='same')(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 2nd conv
    x = Conv2D(filters, kernel_size, padding='same')(x)

    # Adjust shortcut if shape or depth changes
    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1),
                          strides=(2, 2) if downsample else (1, 1),
                          padding='same')(shortcut)

    # Residual merge
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_TDnirsResNet_model(input_shape=(8, 1000, 1),
                       dropout_rate: float = 0.0,
                       output_units: int = 1,
                       compile_model: bool = True) -> Model:
    """Return a compiled Keras model matching the architecture in the prompt."""
    inp = Input(shape=input_shape)

    # Stem
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual stages
    x = residual_block(x, 128, downsample=True, dropout_rate=dropout_rate)
    x = residual_block(x, 128)

    x = residual_block(x, 128, downsample=True, dropout_rate=dropout_rate)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True, dropout_rate=dropout_rate)
    x = residual_block(x, 256)

    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(output_units)(x)

    model = Model(inp, out, name='CustomResNet')

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss='mse',
            metrics=['mae']
        )

    return model