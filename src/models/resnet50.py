import tensorflow as tf

# =========================================================
# RESNET50 MODEL
# =========================================================

def build_resnet50(input_shape, num_classes, dropout_rate=0.4):

    base_model = tf.keras.applications.ResNet50(

        weights="imagenet",

        include_top=False,

        input_shape=input_shape,

        name="resnet50_backbone"

    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = tf.keras.layers.Lambda(
        lambda image: tf.keras.applications.resnet50.preprocess_input(image * 255.0),
        output_shape=input_shape,
        name="resnet50_preprocessing"
    )(inputs)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        512,
        kernel_regularizer=tf.keras.regularizers.l2(2e-4)
    )(x)

    x = tf.keras.layers.Activation("swish")(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model
