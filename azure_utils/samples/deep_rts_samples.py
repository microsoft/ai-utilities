"""
AI-Utilities - deep_rts_samples.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import inspect
import os
import warnings

import keras.backend as K
import numpy as np
from azure_utils.machine_learning.models.training_arg_parsers import get_model_path, process_request, \
    prepare_response, \
    default_response
from azure_utils.resnet152 import RTSEstimator
from azureml.contrib.services import rawhttp
from keras import initializers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import add
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory

WEIGHTS_PATH = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5'


class Scale(Layer):
    """Custom Layer for ResNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

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

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma' % self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta' % self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResNet152(RTSEstimator):
    def __init__(self):
        self.model = None
        print("Created")

    def train(self):
        self.model = self.create_model()
        return self.model

    def save_model(self, path):
        self.model.save_weights(path)

    @classmethod
    def load_model(cls, model_file="/model.pkl"):
        resnet_152 = ResNet152()
        model = resnet_152.train()
        model.load_weights(get_model_path(model_file))
        return resnet_152

    def predict(self, request):
        img_array, transformed_dict = process_request(request)
        preds = self.model.predict(img_array)
        return prepare_response(preds, transformed_dict)

    @staticmethod
    def _identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity_block is the block that has no conv layer at shortcut

        Keyword arguments
        input_tensor -- input tensor
        kernel_size -- defualt 3, the kernel size of middle conv layer at main path
        filters -- list of integers, the nb_filters of 3 conv layer at main path
        stage -- integer, current stage label, used for generating layer names
        block -- 'a','b'..., current block label, used for generating layer names

        """
        eps = 1.1e-5

        if K.image_dim_ordering() == 'tf':
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

    @staticmethod
    def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
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

        if K.image_dim_ordering() == 'tf':
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
        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                   name=conv_name_base + '2b', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b_relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                          name=conv_name_base + '1', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
        shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

        x = add([x, shortcut], name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x

    @staticmethod
    def create_model(include_top=True, weights=None, input_tensor=None, input_shape=None, large_input=False,
                     pooling=None, classes=1000):
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
        assert weights not in {'imagenet', None}, """The `weights` argument should be either `None` (random 
        initialization) or `imagenet` (pre-training on ImageNet)."""

        assert weights == 'imagenet' and include_top and classes != 1000, """If using `weights` as imagenet with 
        `include_top` as true, `classes` should be 1000"""

        eps = 1.1e-5

        if large_input:
            img_size = 448
        else:
            img_size = 224

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=img_size,
                                          min_size=197,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)

        img_input = ResNet152.get_image_input(input_shape, input_tensor)

        # handle dimension ordering for different backends
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        layers_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        layers_x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(layers_x)
        layers_x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(layers_x)
        layers_x = Scale(axis=bn_axis, name='scale_conv1')(layers_x)
        layers_x = Activation('relu', name='conv1_relu')(layers_x)
        layers_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(layers_x)

        layers_x = ResNet152._conv_block(layers_x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        layers_x = ResNet152._identity_block(layers_x, 3, [64, 64, 256], stage=2, block='b')
        layers_x = ResNet152._identity_block(layers_x, 3, [64, 64, 256], stage=2, block='c')

        layers_x = ResNet152._conv_block(layers_x, 3, [128, 128, 512], stage=3, block='a')
        for i in range(1, 8):
            layers_x = ResNet152._identity_block(layers_x, 3, [128, 128, 512], stage=3, block='b' + str(i))

        layers_x = ResNet152._conv_block(layers_x, 3, [256, 256, 1024], stage=4, block='a')
        for i in range(1, 36):
            layers_x = ResNet152._identity_block(layers_x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

        layers_x = ResNet152._conv_block(layers_x, 3, [512, 512, 2048], stage=5, block='a')
        layers_x = ResNet152._identity_block(layers_x, 3, [512, 512, 2048], stage=5, block='b')
        layers_x = ResNet152._identity_block(layers_x, 3, [512, 512, 2048], stage=5, block='c')

        if large_input:
            layers_x = AveragePooling2D((14, 14), name='avg_pool')(layers_x)
        else:
            layers_x = AveragePooling2D((7, 7), name='avg_pool')(layers_x)

        layers_x = ResNet152.add_classification_layer(classes, include_top, layers_x, pooling)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, layers_x, name='resnet152')

        ResNet152._load_weights(include_top, model, weights)
        return model

    @staticmethod
    def add_classification_layer(classes, include_top, layers_x, pooling):
        # include classification layer by default, not included for feature extraction
        if include_top:
            layers_x = Flatten()(layers_x)
            layers_x = Dense(classes, activation='softmax', name='fc1000')(layers_x)
        else:
            if pooling == 'avg':
                layers_x = GlobalAveragePooling2D()(layers_x)
            elif pooling == 'max':
                layers_x = GlobalMaxPooling2D()(layers_x)
        return layers_x

    @staticmethod
    def get_image_input(input_shape, input_tensor):
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        return img_input

    @staticmethod
    def _load_weights(include_top, model, weights):
        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('resnet152_weights_tf.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='cdb18a2158b88e392c0905d47dcef965')
            else:
                weights_path = get_file('resnet152_weights_tf_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='4a90dcdafacbd17d772af1fb44fc2660')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(model)
                if include_top:
                    maxpool = model.get_layer(name='avg_pool')
                    shape = maxpool.output_shape[1:]
                    dense = model.get_layer(name='fc1000')
                    layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')


model = ResNet152.create_model(include_top=True, weights='imagenet')
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

if __name__ == '__main__':
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))


class RealTimeDeepFactory(RealTimeFactory):
    def train(self, args):
        resnet_152 = ResNet152.create_model(weights="imagenet")
        resnet_152.save_model(os.path.join(args.outputs, args.model))

    def score_init(self):
        self.resnet_152 = ResNet152.load_model()

    @rawhttp
    def score_run(self, request):
        """ Make a prediction based on the data passed in using the preloaded model"""
        if request.method == 'POST':
            return self.resnet_152.predict(request)
        return default_response(request)

    def __init__(self):
        self.resnet_152 = None
        print("")

    @staticmethod
    def make_file():
        file = inspect.getsource(RealTimeFactory)
        file = file.replace('def train(self, args):\n        raise NotImplementedError\n',
                            inspect.getsource(RealTimeDeepFactory.train))
        file = file.replace('def score_init(self):\n        raise NotImplementedError\n',
                            inspect.getsource(RealTimeDeepFactory.score_init))
        file = file.replace('@rawhttp\n    def score_run(self, request):\n        raise NotImplementedError\n',
                            inspect.getsource(RealTimeDeepFactory.score_run))
        file = file.replace('def __init__(self):\n        raise NotImplementedError\n',
                            inspect.getsource(RealTimeDeepFactory.__init__))
        file
