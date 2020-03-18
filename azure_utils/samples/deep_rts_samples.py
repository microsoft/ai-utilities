import argparse
import inspect
import io
import os
import sys
import warnings
from typing import Any, Tuple
import keras.backend as keras_backend

import lightgbm as lgb
import numpy as np
import pandas as pd
from azureml.contrib.services import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core import Run
from keras import initializers
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.engine import InputSpec, Layer
from keras.engine.topology import get_source_inputs
from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
    add,
)
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
import joblib
from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory

from azure_utils.machine_learning.item_selector import ItemSelector
from azure_utils.machine_learning.label_rank import label_rank
from azure_utils.machine_learning.training_arg_parsers import (
    default_response,
    get_model_path,
    prepare_response,
    process_request,
)
from azure_utils.rts_estimator import RTSEstimator

imagenet = "imagenet"

weights_path_ = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5"

weights_path_no_top_ = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5"


def get_default_shape(
    data_format: str, default_size, input_shape: Tuple[int, int, int], weights: str
) -> Tuple:
    """
    Get the default shape for validation

    :param data_format: ex: 'channels_first'
    :param default_size: Default Three Int Tuple
    :param input_shape: Three Int Tuple
    :param weights: ex: IMAGENET_
    :return:
    """
    if weights != imagenet and input_shape and len(input_shape) == 3:
        if data_format == "channels_first":
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    f"This model usually expects 1 or 3 input channels. However, it was passed an input_shape "
                    f"with {str(input_shape[0])} input channels."
                )
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    f"This model usually expects 1 or 3 input channels. However, it was passed an input_shape "
                    f"with {str(input_shape[-1])} input channels."
                )
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == "channels_first":
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    return default_shape


"""
AI-Utilities - deep_rts_samples.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""


def make_file():
    """
    Write this class to a string.
    """
    file = inspect.getsource(RealTimeFactory)
    file = file.replace(
        "def train(self, args):\n        raise NotImplementedError\n",
        inspect.getsource(RealTimeDeepFactory.train),
    )
    file = file.replace(
        "def score_init(self):\n        raise NotImplementedError\n",
        inspect.getsource(RealTimeDeepFactory.score_init),
    )
    file = file.replace(
        "@rawhttp\n    def score_run(self, request):\n        raise NotImplementedError\n",
        inspect.getsource(RealTimeDeepFactory.score_run),
    )
    file = file.replace(
        "def __init__(self):\n        raise NotImplementedError\n",
        inspect.getsource(RealTimeDeepFactory.__init__),
    )
    return file


def _obtain_input_shape(
    input_shape, default_size, min_size, data_format, require_flatten, weights=None
):
    """Internal utility to compute/validate a model's input shape.

    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or IMAGENET_ (pre-training on ImageNet).
            If weights=IMAGENET_ input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: In case of invalid argument values.
    """
    default_shape = get_default_shape(data_format, default_size, input_shape, weights)
    if weights == imagenet and require_flatten:
        assert_same_shape(default_shape, input_shape)
        return default_shape
    if input_shape:
        validate_input_shape(data_format, input_shape, min_size, weights)
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == "channels_first":
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten and None in input_shape:
        raise ValueError(
            "If `include_top` is True, "
            "you should specify a static `input_shape`. "
            "Got `input_shape=" + str(input_shape) + "`"
        )
    return input_shape


def validate_input_shape(
    data_format: str, input_shape: Tuple[int, int, int], min_size: int, weights: str
):
    """ Validate the shape of the input

    :param data_format: ex:  'channels_first'
    :param input_shape: Tuple
    :param min_size: minimum size of the shape
    :param weights: ex: imagenet
    """
    if data_format == "channels_first":
        if input_shape is not None:
            assert_three_int_tuple(input_shape)
            if input_shape[0] != 3 and weights == imagenet:
                raise channel_error(input_shape)
            assert_input_size(input_shape, min_size, 1, 2)
    else:
        if input_shape is not None:
            assert_three_int_tuple(input_shape)
            if input_shape[-1] != 3 and weights == imagenet:
                raise channel_error(input_shape)
            assert_input_size(input_shape, min_size, 0, 1)


def assert_same_shape(
    default_shape: Tuple[int, int, int], input_shape: Tuple[int, int, int]
):
    """
    Assert the input shape is the same the as the default shape

    :param default_shape: Default Three Int Tuple to validate with
    :param input_shape: Three Int Tuple to assert
    """
    if input_shape is not None and input_shape != default_shape:
        raise ValueError(
            f"When setting`include_top=True` and loading `imagenet` weights, `input_shape` should be "
            f"{str(default_shape)}."
        )


def channel_error(input_shape: Tuple[int, int, int]) -> ValueError:
    """
    Value Error Message for Channel Error
    :param input_shape: Three Int Tuple
    :return: Custom text Value Error
    """
    return ValueError(
        f"The input must have 3 channels; got `input_shape={str(input_shape)}`"
    )


def assert_three_int_tuple(input_shape: Tuple[int, int, int]):
    """
    Assert Tuple has 3 objects
    :param input_shape: Three Int Tuple
    """
    if len(input_shape) != 3:
        raise ValueError("`input_shape` must be a tuple of three integers.")


def assert_input_size(
    input_shape: Tuple[int, int, int],
    min_size: int,
    first_index: int,
    second_index: int,
):
    """
    Assert that the Input Shape size is above minimum.
    :param input_shape: Three Int Tuple
    :param min_size: Minimum size of Input Shape
    :param first_index: First Tuple Index to test
    :param second_index: Second Tuple Index to test against
    """
    if check_shape_by_index(first_index, input_shape, min_size) or check_shape_by_index(
        second_index, input_shape, min_size
    ):
        raise ValueError(
            f"Input size must be at least {str(min_size)}x{str(min_size)}; got `input_shape={str(input_shape)}`"
        )


def check_shape_by_index(index, input_shape, min_size) -> bool:
    """
    Check the Shape of one object of the tuple.

    :param index: Index of Tuple to Test
    :param input_shape: Input Tuple to test
    :param min_size: Minimum size of of tuple object
    :return: 'bool' result of test
    """
    return input_shape[index] is not None and input_shape[index] < min_size


class Scale(Layer):
    """Custom Layer for ResNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases learned.

    Keyword arguments:
    axis -- integer, axis along which to normalize in mode 0. For instance,
        if your input tensor has shape (samples, channels, rows, cols),
        set axis to 1 to normalize per feature map (channels axis).
    momentum -- momentum in the computation of the exponential average
        of the mean and standard deviation of the data, for
        feature-wise normalization.
    weights -- Initialization weights.
        List of 2 Numpy arrays, with shapes:
        `[(input_shape,), (input_shape,)]`
    beta_init -- name of initialization function for shift parameter
        (see [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
    gamma_init -- name of initialization function for scale parameter (see
        [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.

    """

    def __init__(
        self,
        weights: Any = None,
        axis: int = -1,
        momentum: float = 0.9,
        beta_init: str = "zero",
        gamma_init: str = "one",
        **kwargs,
    ):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights

        self.gamma = None
        self.beta = None
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        """

        :param input_shape:
        """
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = keras_backend.variable(
            self.gamma_init(shape), name="%s_gamma" % self.name
        )
        self.beta = keras_backend.variable(
            self.beta_init(shape), name="%s_beta" % self.name
        )
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, layers, **kwargs):
        """

        :param layers:
        :param kwargs:
        :return:
        """
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = keras_backend.reshape(
            self.gamma, broadcast_shape
        ) * layers + keras_backend.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self) -> dict:
        """

        :return:
        """
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return {**base_config, **config}


class ResNet152(RTSEstimator):
    """
    ResNet-152 Wrapped for Azure Machine Learning
    """

    def __init__(self):
        self.model = None

    def train(self):
        """

        :return:
        """
        self.model = self.create_model()
        return self.model

    def save_model(self, path="model.pkl"):
        """

        :param path:
        """
        self.model.save_weights(path)

    @classmethod
    def load_model(cls, model_file="outputs/model.pkl"):
        """

        :param model_file:
        :return:
        """
        resnet_152 = ResNet152()

        resnet_152.train().load_weights(get_model_path("outputs/model.pkl"))
        return resnet_152

    def predict(self, request):
        """

        :param request:
        :return:
        """
        img_array, transformed_dict = process_request(request)
        predictions = self.model.predict(img_array)
        return prepare_response(predictions, transformed_dict)

    @staticmethod
    def _identity_block(
        input_tensor, kernel_size, filters, stage, block, strides=(1, 1)
    ):
        """The identity_block is the block that has no conv layer at shortcut

            Keyword arguments
            input_tensor -- input tensor
            kernel_size -- defualt 3, the kernel size of middle conv layer at main path
            filters -- list of integers, the nb_filters of 3 conv layer at main path
            stage -- integer, current stage label, used for generating layer names
            block -- 'a','b'..., current block label, used for generating layer names

            """
        eps = 1.1e-5

        if keras_backend.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
        x = Activation('relu', name=conv_name_base + '2a_relu')(x)

        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b_relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

        x = add([x, input_tensor], name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x

    @classmethod
    def _conv_block(
        cls, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
    ):
        """conv_block is the block that has a conv layer at shortcut

            Keyword arguments:
            input_tensor -- input tensor
            kernel_size -- defualt 3, the kernel size of middle conv layer at main path
            filters -- list of integers, the nb_filters of 3 conv layer at main path
            stage -- integer, current stage label, used for generating layer names
            block -- 'a','b'..., current block label, used for generating layer names

            Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
            And the shortcut should have subsample=(2,2) as well

            """
        eps = 1.1e-5

        if keras_backend.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
        x = Activation('relu', name=conv_name_base + '2a_relu')(x)

        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b_relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
        shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

        x = add([x, shortcut], name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x

    def create_model(
        self,
        include_top: bool = True,
        weights: str = None,
        input_tensor: Any = None,
        input_shape: Any = None,
        large_input: bool = False,
        pooling: Any = None,
        classes: int = 1000,
        save_model: bool = False,
        model_path: str = None,
    ) -> Model:
        """Instantiate the ResNet152 architecture.

        Keyword arguments:
        include_top -- whether to include the fully-connected layer at the
            top of the network. (default True)
        weights -- one of `None` (random initialization) or "imagenet"
            (pre-training on ImageNet). (default None)
        input_tensor -- optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.(default None)
        input_shape -- optional shape tuple, only to be specified if
            `include_top` is False (otherwise the input shape has to be
            `(224, 224, 3)` (with `channels_last` data format) or
            `(3, 224, 224)` (with `channels_first` data format). It should
            have exactly 3 inputs channels, and width and height should be
            no smaller than 197. E.g. `(200, 200, 3)` would be one valid value.
            (default None)
        large_input -- if True, then the input shape expected will be
            `(448, 448, 3)` (with `channels_last` data format) or
            `(3, 448, 448)` (with `channels_first` data format). (default False)
        pooling -- Optional pooling mode for feature extraction when
            `include_top` is `False`.
            - `None` means that the output of the model will be the 4D
                tensor output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to
                the output of the last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
            (default None)
        classes -- optional number of classes to classify image into, only
            to be specified if `include_top` is True, and if no `weights`
            argument is specified. (default 1000)

        Returns:
        A Keras model instance.

        Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
            :param include_top:
            :param weights:
            :param input_tensor:
            :param input_shape:
            :param large_input:
            :param pooling:
            :param classes:
            :param save_model:
            :param model_path:
            :return:
        """
        assert weights in {
            imagenet,
            None,
        }, """The `weights` argument should be either `None` (random 
        initialization) or `imagenet` (pre-training on ImageNet)."""
        if weights == imagenet and include_top:
            assert (
                 classes == 1000
            ), """If using `weights` as imagenet with 
            `include_top` as true, `classes` should be 1000"""

        eps = 1.1e-5

        if large_input:
            img_size = 448
        else:
            img_size = 224

        def _obtain_input_shape(
            input_shape,
            default_size,
            min_size,
            data_format,
            require_flatten,
            weights=None,
        ):
            """Internal utility to compute/validate a model's input shape.

            # Arguments
                input_shape: Either None (will return the default network input shape),
                    or a user-provided shape to be validated.
                default_size: Default input width/height for the model.
                min_size: Minimum input width/height accepted by the model.
                data_format: Image data format to use.
                require_flatten: Whether the model is expected to
                    be linked to a classifier via a Flatten layer.
                weights: One of `None` (random initialization)
                    or IMAGENET_ (pre-training on ImageNet).
                    If weights=IMAGENET_ input channels must be equal to 3.

            # Returns
                An integer shape tuple (may include None entries).

            # Raises
                ValueError: In case of invalid argument values.
            """
            default_shape = get_default_shape(
                data_format, default_size, input_shape, weights
            )
            if weights == imagenet and require_flatten:
                assert_same_shape(default_shape, input_shape)
                return default_shape
            if input_shape:
                validate_input_shape(data_format, input_shape, min_size, weights)
            else:
                if require_flatten:
                    input_shape = default_shape
                else:
                    if data_format == "channels_first":
                        input_shape = (3, None, None)
                    else:
                        input_shape = (None, None, 3)
            if require_flatten and None in input_shape:
                raise ValueError(
                    "If `include_top` is True, "
                    "you should specify a static `input_shape`. "
                    "Got `input_shape=" + str(input_shape) + "`"
                )
            return input_shape

        # Determine proper input shape
        input_shape = _obtain_input_shape(
            input_shape,
            default_size=img_size,
            min_size=197,
            data_format=keras_backend.image_data_format(),
            require_flatten=include_top,
        )

        img_input = ResNet152.get_image_input(input_shape, input_tensor)

        # handle dimension ordering for different backends
        if keras_backend.common.image_dim_ordering() == "tf":
            bn_axis = 3
        else:
            bn_axis = 1

        layers_x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(img_input)
        layers_x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=False)(
            layers_x
        )
        layers_x = BatchNormalization(epsilon=eps, axis=bn_axis, name="bn_conv1")(
            layers_x
        )
        layers_x = Scale(axis=bn_axis, name="scale_conv1")(layers_x)
        layers_x = Activation("relu", name="conv1_relu")(layers_x)
        layers_x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(layers_x)

        layers_x = ResNet152._conv_block(
            layers_x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1)
        )
        layers_x = ResNet152._identity_block(
            layers_x, 3, [64, 64, 256], stage=2, block="b"
        )
        layers_x = ResNet152._identity_block(
            layers_x, 3, [64, 64, 256], stage=2, block="c"
        )

        layers_x = ResNet152._conv_block(
            layers_x, 3, [128, 128, 512], stage=3, block="a"
        )
        for i in range(1, 8):
            layers_x = ResNet152._identity_block(
                layers_x, 3, [128, 128, 512], stage=3, block="b" + str(i)
            )

        layers_x = ResNet152._conv_block(
            layers_x, 3, [256, 256, 1024], stage=4, block="a"
        )
        for i in range(1, 36):
            layers_x = ResNet152._identity_block(
                layers_x, 3, [256, 256, 1024], stage=4, block="b" + str(i)
            )

        layers_x = ResNet152._conv_block(
            layers_x, 3, [512, 512, 2048], stage=5, block="a"
        )
        layers_x = ResNet152._identity_block(
            layers_x, 3, [512, 512, 2048], stage=5, block="b"
        )
        layers_x = ResNet152._identity_block(
            layers_x, 3, [512, 512, 2048], stage=5, block="c"
        )

        if large_input:
            layers_x = AveragePooling2D((14, 14), name="avg_pool")(layers_x)
        else:
            layers_x = AveragePooling2D((7, 7), name="avg_pool")(layers_x)

        layers_x = ResNet152.add_classification_layer(
            classes, include_top, layers_x, pooling
        )

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, layers_x, name="resnet152")

        ResNet152._load_weights(include_top, model, weights)
        self.model = model

        if save_model:
            self.save_model(model_path)

        return model

    @staticmethod
    def add_classification_layer(classes, include_top, layers_x, pooling):
        """

        :param classes:
        :param include_top:
        :param layers_x:
        :param pooling:
        :return:
        """
        # include classification layer by default, not included for feature extraction
        if include_top:
            layers_x = Flatten()(layers_x)
            layers_x = Dense(classes, activation="softmax", name="fc1000")(layers_x)
        else:
            if pooling == "avg":
                layers_x = GlobalAveragePooling2D()(layers_x)
            elif pooling == "max":
                layers_x = GlobalMaxPooling2D()(layers_x)
        return layers_x

    @staticmethod
    def get_image_input(input_shape, input_tensor):
        """

        :param input_shape:
        :param input_tensor:
        :return:
        """
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not keras_backend.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        return img_input

    @staticmethod
    def _load_weights(include_top, model, weights):
        # load weights
        if weights == imagenet:
            if include_top:
                weights_path = get_file(
                    "resnet152_weights_tf.h5",
                    weights_path_,
                    cache_subdir="models",
                    md5_hash="cdb18a2158b88e392c0905d47dcef965",
                )
            else:
                weights_path = get_file(
                    "resnet152_weights_tf_notop.h5",
                    weights_path_no_top_,
                    cache_subdir="models",
                    md5_hash="4a90dcdafacbd17d772af1fb44fc2660",
                )
            model.load_weights(weights_path, by_name=True)
            if keras_backend.backend() == "theano":
                layer_utils.convert_all_kernels_in_model(model)
                if include_top:
                    maxpool = model.get_layer(name="avg_pool")
                    shape = maxpool.output_shape[1:]
                    dense = model.get_layer(name="fc1000")
                    layer_utils.convert_dense_weights_data_format(dense, shape)

            if (
                keras_backend.image_data_format() == "channels_first"
                and keras_backend.backend() == "tensorflow"
            ):
                warnings.warn(
                    "You are using the TensorFlow backend, yet you "
                    "are using the Theano "
                    "image data format convention "
                    '(`image_data_format="channels_first"`). '
                    "For best performance, set "
                    '`image_data_format="channels_last"` in '
                    "your Keras config "
                    "at ~/.keras/keras.json."
                )


class RealTimeDeepFactory(RealTimeFactory):

    """
    Factory to create Real-time Scoring service with Azure Machine Learning
    """

    def train(self, args):
        """

        :param args:
        """
        self.trained_model = ResNet152().create_model(
            weights="imagenet",
            save_model=True,
            model_path=os.path.join(args.outputs, args.model),
        )

    def score_init(self):
        """
        Start up function for Scoring Applet
        """
        self.scoring_model = ResNet152.load_model()

    @rawhttp
    def score_run(self, request) -> AMLResponse:
        """ Make a prediction based on the data passed in using the preloaded model"""
        if request.method == "POST":
            return self.scoring_model.predict(request)
        return default_response(request)

    def __init__(self):
        """
        Create Training and Scoring Models
        """
        super().__init__()
        self.trained_model = None
        self.scoring_model = None


def MakeResNet152(include_top=True, weights=None, input_tensor=None, input_shape=None, large_input=False, pooling=None,
              classes=1000):
    """Instantiate the ResNet152 architecture.

    Keyword arguments:
    include_top -- whether to include the fully-connected layer at the
        top of the network. (default True)
    weights -- one of `None` (random initialization) or "imagenet"
        (pre-training on ImageNet). (default None)
    input_tensor -- optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.(default None)
    input_shape -- optional shape tuple, only to be specified if
        `include_top` is False (otherwise the input shape has to be
        `(224, 224, 3)` (with `channels_last` data format) or
        `(3, 224, 224)` (with `channels_first` data format). It should
        have exactly 3 inputs channels, and width and height should be
        no smaller than 197. E.g. `(200, 200, 3)` would be one valid value.
        (default None)
    large_input -- if True, then the input shape expected will be
        `(448, 448, 3)` (with `channels_last` data format) or
        `(3, 448, 448)` (with `channels_first` data format). (default False)
    pooling -- Optional pooling mode for feature extraction when
        `include_top` is `False`.
        - `None` means that the output of the model will be the 4D
            tensor output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to
            the output of the last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
        (default None)
    classes -- optional number of classes to classify image into, only
        to be specified if `include_top` is True, and if no `weights`
        argument is specified. (default 1000)

    Returns:
    A Keras model instance.

    Raises:
    ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    print("Creating Tensorflow model (and hiding Tensorflow output)")

    eps = 1.1e-5

    if large_input:
        img_size = 448
    else:
        img_size = 224

    def _obtain_input_shape(input_shape, default_size, min_size, data_format, require_flatten, weights=None, ):
        """Internal utility to compute/validate a model's input shape.

        # Arguments
            input_shape: Either None (will return the default network input shape),
                or a user-provided shape to be validated.
            default_size: Default input width/height for the model.
            min_size: Minimum input width/height accepted by the model.
            data_format: Image data format to use.
            require_flatten: Whether the model is expected to
                be linked to a classifier via a Flatten layer.
            weights: One of `None` (random initialization)
                or IMAGENET_ (pre-training on ImageNet).
                If weights=IMAGENET_ input channels must be equal to 3.

        # Returns
            An integer shape tuple (may include None entries).

        # Raises
            ValueError: In case of invalid argument values.
        """

        def get_default_shape(data_format: str, default_size, input_shape: Tuple[int, int, int], weights: str) -> Tuple:
            """
            Get the default shape for validation

            :param data_format: ex: 'channels_first'
            :param default_size: Default Three Int Tuple
            :param input_shape: Three Int Tuple
            :param weights: ex: IMAGENET_
            :return:
            """
            if weights != imagenet and input_shape and len(input_shape) == 3:
                if data_format == "channels_first":
                    if input_shape[0] not in {1, 3}:
                        warnings.warn(
                            f"This model usually expects 1 or 3 input channels. However, it was passed an input_shape "
                            f"with {str(input_shape[0])} input channels.")
                    default_shape = (input_shape[0], default_size, default_size)
                else:
                    if input_shape[-1] not in {1, 3}:
                        warnings.warn(
                            f"This model usually expects 1 or 3 input channels. However, it was passed an input_shape "
                            f"with {str(input_shape[-1])} input channels.")
                    default_shape = (default_size, default_size, input_shape[-1])
            else:
                if data_format == "channels_first":
                    default_shape = (3, default_size, default_size)
                else:
                    default_shape = (default_size, default_size, 3)
            return default_shape

        default_shape = get_default_shape(data_format, default_size, input_shape, weights)
        if weights == imagenet and require_flatten:
            assert_same_shape(default_shape, input_shape)
            return default_shape
        if input_shape:
            validate_input_shape(data_format, input_shape, min_size, weights)
        else:
            if require_flatten:
                input_shape = default_shape
            else:
                if data_format == "channels_first":
                    input_shape = (3, None, None)
                else:
                    input_shape = (None, None, 3)
        if require_flatten and None in input_shape:
            raise ValueError("If `include_top` is True, "
                             "you should specify a static `input_shape`. "
                             "Got `input_shape=" + str(input_shape) + "`")
        return input_shape

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=img_size, min_size=197,
                                      data_format=keras_backend.image_data_format(), require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not keras_backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # handle dimension ordering for different backends
    if keras_backend.common.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = ResNet152._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = ResNet152._identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = ResNet152._identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = ResNet152._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = ResNet152._identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = ResNet152._conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = ResNet152._identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = ResNet152._conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = ResNet152._identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = ResNet152._identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if large_input:
        x = AveragePooling2D((14, 14), name='avg_pool')(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # include classification layer by default, not included for feature extraction
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet152')
    # load weights
    WEIGHTS_PATH = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5'

    if weights == "imagenet":
        if include_top:
            weights_path = get_file("resnet152_weights_tf.h5", WEIGHTS_PATH, cache_subdir="models",
                md5_hash="cdb18a2158b88e392c0905d47dcef965", )
        else:
            weights_path = get_file("resnet152_weights_tf_notop.h5", WEIGHTS_PATH_NO_TOP, cache_subdir="models",
                md5_hash="4a90dcdafacbd17d772af1fb44fc2660", )
        model.load_weights(weights_path, by_name=True)
        if keras_backend.backend() == "theano":
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name="avg_pool")
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name="fc1000")
                layer_utils.convert_dense_weights_data_format(dense, shape, "channels_first")

        if keras_backend.image_data_format() == "channels_first" and keras_backend.backend() == "tensorflow":
            warnings.warn("You are using the TensorFlow backend, yet you "
                          "are using the Theano "
                          "image data format convention "
                          '(`image_data_format="channels_first"`). '
                          "For best performance, set "
                          '`image_data_format="channels_last"` in '
                          "your Keras config "
                          "at ~/.keras/keras.json.")
    return model

_NUMBER_RESULTS = 3
from toolz import compose
from PIL import Image, ImageOps

def _image_ref_to_pil_image(image_ref):
    """ Load image with PIL (RGB)
    """
    return Image.open(image_ref).convert("RGB")


def _pil_to_numpy(pil_image):
    img = ImageOps.fit(pil_image, (224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img

def get_model_api():
    import logging
    import timeit as t
    logger = logging.getLogger("model_driver")

    def _create_scoring_func():
        """ Initialize ResNet 152 Model
        """
        logger = logging.getLogger("model_driver")
        start = t.default_timer()
        model = MakeResNet152(weights="imagenet")
        # model.load_weights("outputs/model.pkl")
        end = t.default_timer()

        loadTimeMsg = "Model loading time: {0} ms".format(round((end - start) * 1000, 2))
        logger.info(loadTimeMsg)

        def call_model(img_array_list):
            img_array = np.stack(img_array_list)
            img_array = preprocess_input(img_array)
            preds = model.predict(img_array)
            # Converting predictions to float64 since we are able to serialize float64 but not float32
            preds = decode_predictions(preds.astype(np.float64), top=_NUMBER_RESULTS)
            return preds

        return call_model

    scoring_func = _create_scoring_func()

    def process_and_score(images_dict):
        """ Classify the input using the loaded model
        """
        start = t.default_timer()
        logger.info("Scoring {} images".format(len(images_dict)))
        transform_input = compose(_pil_to_numpy, _image_ref_to_pil_image)
        transformed_dict = {key: transform_input(img_ref) for key, img_ref in images_dict.items()}
        preds = scoring_func(list(transformed_dict.values()))
        preds = dict(zip(transformed_dict.keys(), preds))
        end = t.default_timer()

        logger.info("Predictions: {0}".format(preds))
        logger.info("Predictions took {0} ms".format(round((end - start) * 1000, 2)))
        return (preds, "Computed in {0} ms".format(round((end - start) * 1000, 2)))

    return process_and_score


def main():
    """ Main Method to use with AzureML"""
    # Define the arguments.
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a model based on train-test datasets."
    )
    parser.add_argument(
        "-d",
        "--train_data",
        help="the training dataset name",
        default="balanced_pairs_train.tsv",
    )
    parser.add_argument(
        "-t",
        "--test_data",
        help="the test dataset name",
        default="balanced_pairs_test.tsv",
    )
    parser.add_argument(
        "-i",
        "--estimators",
        help="the number of learner estimators",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--min_child_samples",
        help="the minimum number of samples in a child(leaf)",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-v", "--verbose", help="the verbosity of the estimator", type=int, default=-1
    )
    parser.add_argument(
        "-n", "--ngrams", help="the maximum size of word ngrams", type=int, default=1
    )
    parser.add_argument(
        "-u",
        "--unweighted",
        help="do not use instance weights",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--match",
        help="the maximum number of duplicate matches",
        type=int,
        default=20,
    )
    parser.add_argument("--outputs", help="the outputs directory", default=".")
    parser.add_argument("--inputs", help="the inputs directory", default=".")
    parser.add_argument(
        "-s", "--save", help="save the model", action="store_true", default=True
    )
    parser.add_argument("--model", help="the model file", default="model.pkl")
    parser.add_argument("--instances", help="the instances file", default="inst.txt")
    parser.add_argument("--labels", help="the labels file", default="labels.txt")
    parser.add_argument(
        "-r", "--rank", help="the maximum rank of correct answers", type=int, default=3
    )
    args = parser.parse_args()

    run = Run.get_context()

    # The training and testing datasets.
    inputs_path = args.inputs
    data_path = os.path.join(inputs_path, args.train_data)
    test_path = os.path.join(inputs_path, args.test_data)

    # Create the outputs folder.
    outputs_path = args.outputs
    os.makedirs(outputs_path, exist_ok=True)
    model_path = os.path.join(outputs_path, args.model)
    instances_path = os.path.join(outputs_path, args.instances)
    labels_path = os.path.join(outputs_path, args.labels)

    # Load the training data
    print("Reading {}".format(data_path))
    train = pd.read_csv(data_path, sep="\t", encoding="latin1")

    # Limit the number of duplicate-original question matches.
    train = train[train.n < args.match]

    # Define the roles of the columns in the training data.
    feature_columns = ["Text_x", "Text_y"]
    label_column = "Label"
    duplicates_id_column = "Id_x"
    answer_id_column = "AnswerId_y"

    # Report on the training dataset: the number of rows and the proportion of true matches.
    print(
        "train: {:,} rows with {:.2%} matches".format(
            train.shape[0], train[label_column].mean()
        )
    )

    # Compute the instance weights used to correct for class imbalance in training.
    weight_column = "Weight"
    if args.unweighted:
        weight = pd.Series([1.0], train[label_column].unique())
    else:
        label_counts = train[label_column].value_counts()
        weight = train.shape[0] / (label_counts.shape[0] * label_counts)
    train[weight_column] = train[label_column].apply(lambda x: weight[x])

    # Collect the unique ids that identify each original question's answer.
    labels = sorted(train[answer_id_column].unique())
    label_order = pd.DataFrame({"label": labels})

    # Collect the parts of the training data by role.
    train_x = train[feature_columns]
    train_y = train[label_column]
    sample_weight = train[weight_column]

    # Use the inputs to define the hyperparameters used in training.
    n_estimators = args.estimators
    min_child_samples = args.min_child_samples
    if args.ngrams > 0:
        ngram_range = (1, args.ngrams)
    else:
        ngram_range = None

    # Verify that the hyperparameter values are valid.
    assert n_estimators > 0
    assert min_child_samples > 1
    assert isinstance(ngram_range, tuple) and len(ngram_range) == 2
    assert 0 < ngram_range[0] <= ngram_range[1]

    # Define the pipeline that featurizes the text columns.
    featurization = [
        (
            column,
            make_pipeline(
                ItemSelector(column), text.TfidfVectorizer(ngram_range=ngram_range)
            ),
        )
        for column in feature_columns
    ]
    features = FeatureUnion(featurization)

    # Define the estimator that learns how to classify duplicate-original question pairs.
    estimator = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        verbose=args.verbose,
    )

    # Define the model pipeline as feeding the features into the estimator.
    model = Pipeline([("features", features), ("model", estimator)])

    # Fit the model.
    print("Training...")
    model.fit(train_x, train_y, model__sample_weight=sample_weight)

    # Save the model to a file, and report on its size.
    if args.save:
        joblib.dump(model, model_path)
        print(
            "{} size: {:.2f} MB".format(
                model_path, os.path.getsize(model_path) / (2 ** 20)
            )
        )

    # Test the model
    # Read in the test data set, and report of the number of its rows and proportion of true matches.
    print("Reading {}".format(test_path))
    test = pd.read_csv(test_path, sep="\t", encoding="latin1")
    print(
        "test: {:,} rows with {:.2%} matches".format(
            test.shape[0], test[label_column].mean()
        )
    )

    # Collect the model predictions. This step should take about 1 minute on a Standard NC6 DLVM.
    print("Testing...")
    test_x = test[feature_columns]
    test["probabilities"] = model.predict_proba(test_x)[:, 1]

    # Collect the probabilities for each duplicate question, ordered by the original question ids.
    # Order the testing data by duplicate question id and original question id.
    test.sort_values([duplicates_id_column, answer_id_column], inplace=True)

    # Extract the ordered probabilities.
    probabilities = test.probabilities.groupby(
        test[duplicates_id_column], sort=False
    ).apply(lambda x: tuple(x.values))

    # Create a data frame with one row per duplicate question, and make it contain the model's predictions for each
    # duplicate.
    test_score = (
        test[["Id_x", "AnswerId_x", "Text_x"]]
        .drop_duplicates()
        .set_index(duplicates_id_column)
    )
    test_score["probabilities"] = probabilities
    test_score.reset_index(inplace=True)
    test_score.columns = ["Id", "AnswerId", "Text", "probabilities"]

    # Evaluate the predictions
    # For each duplicate question, find the rank of its correct original question.
    test_score["Ranks"] = test_score.apply(
        lambda x: label_rank(x.AnswerId, x.probabilities, label_order.label), axis=1
    )

    # Compute the fraction of correct original questions by minimum rank. Also print the average rank of the correct
    # original questions.
    for i in range(1, args.rank + 1):
        print("Accuracy @{} = {:.2%}".format(i, (test_score["Ranks"] <= i).mean()))
        run.log("Accuracy @{}".format(i), (test_score["Ranks"] <= i).mean())
    mean_rank = test_score["Ranks"].mean()
    print("Mean Rank {:.4f}".format(mean_rank))
    run.log("Mean Rank", mean_rank)

    # Write the scored instances to a file, along with the ordered original questions's answer ids.
    test_score.to_csv(instances_path, sep="\t", index=False, encoding="latin1")
    label_order.to_csv(labels_path, sep="\t", index=False)


if __name__ == "__main__":
    resnet_152_model = ResNet152().create_model(weights=imagenet)
    img_path = "elephant.jpg"
    img = image.load_img(img_path, target_size=(224, 224))

    image_array = image.img_to_array(img)

    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    print("Input image shape:", image_array.shape)
    print("Predicted:", decode_predictions(resnet_152_model.predict(image_array)))
