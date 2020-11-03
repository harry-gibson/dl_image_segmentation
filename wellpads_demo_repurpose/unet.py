"""
Implementation of the UNet with depth = 5
"""

import tensorflow as tf


class _Conv(tf.keras.layers.Layer):
    def __init__(self, filters, block, path, batch_norm=False, **conv_kwargs):
        super(_Conv, self).__init__()
        self.batch_norm = batch_norm

        name_conv = "conv_{}_{}".format(path, block)
        name_bn = "bn_{}_{}".format(path, block)

        self.conv_1 = tf.keras.layers.Conv2D(
            filters, name=name_conv + "a", **conv_kwargs
        )
        self.bn_1 = tf.keras.layers.BatchNormalization(name=name_bn + "a")
        self.conv_2 = tf.keras.layers.Conv2D(
            filters, name=name_conv + "b", **conv_kwargs
        )
        self.bn_2 = tf.keras.layers.BatchNormalization(name=name_bn + "b")

    def call(self, inputs, training=True):
        x = self.conv_1(inputs, training=training)
        if self.batch_norm:
            x = self.bn_1(x, training=training)
        x = self.conv_2(x, training=training)
        if self.batch_norm:
            x = self.bn_2(x, training=training)

        return x


class _UpConv(tf.keras.layers.Layer):
    def __init__(self, size, filters, **conv_kwargs):
        super(_UpConv, self).__init__()

        self.upsample = tf.keras.layers.UpSampling2D(
            size=size, interpolation="bilinear"
        )
        self.conv = tf.keras.layers.Conv2D(filters, **conv_kwargs)

    def call(self, inputs, training=True):
        x = self.upsample(inputs, training=training)
        x = self.conv(x, training=training)

        return x


class UNet(tf.keras.Model):
    def __init__(
        self,
        n_classes=1,
        n1_filters=32,
        activation="relu",
        padding="same",
        batch_norm=False,
        include_top=True,
        l2_reg=0.0,
    ):
        super(UNet, self).__init__()
        self.include_top = include_top
        conv_kwargs = dict(
            kernel_size=(3, 3),
            activation=activation,
            padding=padding,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        self.conv_down_1 = _Conv(
            filters=n1_filters,
            block=1,
            path="down",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_down_2 = _Conv(
            filters=n1_filters * 2,
            block=2,
            path="down",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_down_3 = _Conv(
            filters=n1_filters * 4,
            block=3,
            path="down",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_down_4 = _Conv(
            filters=n1_filters * 8,
            block=4,
            path="down",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_down_5 = _Conv(
            filters=n1_filters * 16,
            block=5,
            path="down",
            batch_norm=batch_norm,
            **conv_kwargs
        )

        self.conv_transpose_1 = _UpConv(
            size=(2, 2),
            filters=n1_filters * 8,
            kernel_size=(2, 2),
            padding=padding,
            name="conv_transpose_1",
        )
        self.conv_transpose_2 = _UpConv(
            size=(2, 2),
            filters=n1_filters * 4,
            kernel_size=(2, 2),
            padding=padding,
            name="conv_transpose_2",
        )
        self.conv_transpose_3 = _UpConv(
            size=(2, 2),
            filters=n1_filters * 2,
            kernel_size=(2, 2),
            padding=padding,
            name="conv_transpose_3",
        )
        self.conv_transpose_4 = _UpConv(
            size=(2, 2),
            filters=n1_filters,
            kernel_size=(2, 2),
            padding=padding,
            name="conv_transpose_4",
        )

        self.conv_up_1 = _Conv(
            filters=n1_filters * 8,
            block=1,
            path="up",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_up_2 = _Conv(
            filters=n1_filters * 4,
            block=2,
            path="up",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_up_3 = _Conv(
            filters=n1_filters * 2,
            block=3,
            path="up",
            batch_norm=batch_norm,
            **conv_kwargs
        )
        self.conv_up_4 = _Conv(
            filters=n1_filters, block=1, path="up", batch_norm=batch_norm, **conv_kwargs
        )

        self._output = tf.keras.layers.Conv2D(
            filters=n_classes, kernel_size=(1, 1), activation="sigmoid", name="output"
        )

    def call(self, inputs, training=True):
        conv1 = self.conv_down_1(inputs, training=training)
        conv1_ = tf.keras.layers.MaxPool2D(name="max_pool_1")(conv1)
        conv2 = self.conv_down_2(conv1_, training=training)
        conv2_ = tf.keras.layers.MaxPool2D(name="max_pool_2")(conv2)
        conv3 = self.conv_down_3(conv2_, training=training)
        conv3_ = tf.keras.layers.MaxPool2D(name="max_pool_3")(conv3)
        conv4 = self.conv_down_4(conv3_, training=training)
        conv4_ = tf.keras.layers.MaxPool2D(name="max_pool_4")(conv4)
        conv5 = self.conv_down_5(conv4_, training=training)
        x = self.conv_transpose_1(conv5, training=training)
        x = tf.keras.layers.concatenate([x, conv4], axis=3)
        x = self.conv_up_1(x, training=training)
        x = self.conv_transpose_2(x, training=training)
        x = tf.keras.layers.concatenate([x, conv3], axis=3)
        x = self.conv_up_2(x, training=training)
        x = self.conv_transpose_3(x, training=training)
        x = tf.keras.layers.concatenate([x, conv2], axis=3)
        x = self.conv_up_3(x, training=training)
        x = self.conv_transpose_4(x, training=training)
        x = tf.keras.layers.concatenate([x, conv1], axis=3)
        x = self.conv_up_4(x, training=training)
        if self.include_top:
            x = self._output(x, training=training)

        return x
