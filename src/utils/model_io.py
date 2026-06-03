import json
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf


def _deserialize_dtype(dtype):
    if isinstance(dtype, dict):
        try:
            return tf.keras.utils.deserialize_keras_object(dtype)
        except (TypeError, ValueError):
            return dtype.get("config", {}).get("name", "float32")
    return dtype


@tf.keras.utils.register_keras_serializable(package="CancerPrediction")
class ResNet50Preprocessing(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dtype"] = _deserialize_dtype(kwargs.get("dtype"))
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["dtype"] = _deserialize_dtype(config.get("dtype"))
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CancerPrediction")
class MobileNetV2Preprocessing(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dtype"] = _deserialize_dtype(kwargs.get("dtype"))
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["dtype"] = _deserialize_dtype(config.get("dtype"))
        return cls(**config)


def _inject_tensorflow_into_lambda_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda) and hasattr(layer, "function"):
            try:
                layer.function.__globals__.setdefault("tf", tf)
            except AttributeError:
                pass

        if isinstance(layer, tf.keras.Model):
            _inject_tensorflow_into_lambda_layers(layer)

    return model


def _repair_layer_config(layer):
    layer_config = layer.get("config", {})

    if layer.get("class_name") != "Lambda":
        return False

    dtype = _deserialize_dtype(layer_config.get("dtype", "float32"))
    if hasattr(dtype, "name"):
        dtype = dtype.name

    if layer.get("name") == "resnet50_preprocessing" or layer_config.get("name") == "resnet50_preprocessing":
        layer["module"] = None
        layer["class_name"] = "ResNet50Preprocessing"
        layer["registered_name"] = "CancerPrediction>ResNet50Preprocessing"
        layer["config"] = {
            "name": layer_config.get("name", "resnet50_preprocessing"),
            "trainable": layer_config.get("trainable", True),
            "dtype": dtype,
        }
        return True

    if layer.get("name") == "mobilenetv2_preprocessing" or layer_config.get("name") == "mobilenetv2_preprocessing":
        layer["module"] = None
        layer["class_name"] = "MobileNetV2Preprocessing"
        layer["registered_name"] = "CancerPrediction>MobileNetV2Preprocessing"
        layer["config"] = {
            "name": layer_config.get("name", "mobilenetv2_preprocessing"),
            "trainable": layer_config.get("trainable", True),
            "dtype": dtype,
        }
        return True

    if layer_config.get("output_shape") is None:
        layer_config["output_shape"] = [None, None, 3]
        return True

    return False


def _write_portable_repaired_copy(model_path):
    model_path = Path(model_path)
    repaired_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_portable_repaired.keras"

    with zipfile.ZipFile(model_path, "r") as source, zipfile.ZipFile(
        repaired_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as target:
        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename == "config.json":
                config = json.loads(data)
                changed = False

                for layer in config.get("config", {}).get("layers", []):
                    changed = _repair_layer_config(layer) or changed

                if changed:
                    data = json.dumps(config).encode("utf-8")

            target.writestr(item, data)

    return repaired_path


def load_keras_model(model_path):
    custom_objects = {
        "swish": tf.keras.activations.swish,
        "tf": tf,
        "ResNet50Preprocessing": ResNet50Preprocessing,
        "CancerPrediction>ResNet50Preprocessing": ResNet50Preprocessing,
        "src.utils.model_io.ResNet50Preprocessing": ResNet50Preprocessing,
        "MobileNetV2Preprocessing": MobileNetV2Preprocessing,
        "CancerPrediction>MobileNetV2Preprocessing": MobileNetV2Preprocessing,
        "src.utils.model_io.MobileNetV2Preprocessing": MobileNetV2Preprocessing,
    }

    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects=custom_objects)
        return _inject_tensorflow_into_lambda_layers(model)
    except (NotImplementedError, TypeError, ValueError) as exc:
        error_text = str(exc)
        if (
            "Lambda" not in error_text
            and "output_shape" not in error_text
            and "marshal" not in error_text
            and "bad marshal data" not in error_text
        ):
            raise

        repaired_path = _write_portable_repaired_copy(model_path)
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(repaired_path, safe_mode=False, custom_objects=custom_objects)
        return _inject_tensorflow_into_lambda_layers(model)
