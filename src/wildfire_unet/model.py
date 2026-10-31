"""Attention U-Net for next-day fire-mask prediction."""

from __future__ import annotations

from keras import Model, layers


def conv_block(x, filters: int, kernel_size: int = 3):
    """Two (Conv -> BatchNorm -> ReLU) layers."""
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def attention_gate(x, g, inter_channels: int):
    """Additive attention gate.

    Args:
        x: encoder skip-connection features.
        g: gating signal (up-sampled decoder features) at the same resolution as ``x``.
        inter_channels: width of the intermediate projection.

    Returns:
        ``x`` re-weighted by a per-pixel attention coefficient in [0, 1].
    """
    theta_x = layers.Conv2D(inter_channels, 1, padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, 1, padding="same")(g)
    activated = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(activated)
    return layers.Multiply()([x, psi])


def build_attention_unet(input_shape: tuple[int, int, int] = (64, 64, 12)) -> Model:
    """Four-level Attention U-Net (64-128-256-512 encoder, 1024-channel bridge).

    Output is a single-channel sigmoid map: the probability that each pixel is on fire the next day.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    c2 = conv_block(layers.MaxPooling2D()(c1), 128)
    c3 = conv_block(layers.MaxPooling2D()(c2), 256)
    c4 = conv_block(layers.MaxPooling2D()(c3), 512)

    # Bridge
    bridge = conv_block(layers.MaxPooling2D()(c4), 1024)

    # Decoder: each skip connection is filtered by an attention gate before concatenation
    x = bridge
    for skip, filters in ((c4, 512), (c3, 256), (c2, 128), (c1, 64)):
        up = layers.UpSampling2D()(x)
        gated = attention_gate(skip, up, filters)
        x = conv_block(layers.Concatenate()([up, gated]), filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="attention_unet")
