import os

import pytest
from azureml.core import Model

from azure_utils.configuration.notebook_config import project_configuration_file
from azure_utils.configuration.project_configuration import ProjectConfiguration
from azure_utils.machine_learning.contexts.realtime_score_context import (
    RealtimeScoreAKSContext,
    MLRealtimeScore,
    DeepRealtimeScore,
)
from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from tests.mocks.azureml.azureml_mocks import MockMLRealtimeScore, MockDeepRealtimeScore

DEEP_TRAIN_PY = """

        import keras.backend as K
        from keras import initializers
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
        from keras.utils import layer_utils
        from keras.utils.data_utils import get_file

        WEIGHTS_PATH = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5"
        WEIGHTS_PATH_NO_TOP = "https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5"

        def _obtain_input_shape(input_shape,
                                default_size,
                                min_size,
                                data_format,
                                require_flatten,
                                weights=None):
            if weights != 'imagenet' and input_shape and len(input_shape) == 3:
                if data_format == 'channels_first':
                    if input_shape[0] not in {1, 3}:
                        warnings.warn(
                            'This model usually expects 1 or 3 input channels. '
                            'However, it was passed an input_shape with ' +
                            str(input_shape[0]) + ' input channels.')
                    default_shape = (input_shape[0], default_size, default_size)
                else:
                    if input_shape[-1] not in {1, 3}:
                        warnings.warn(
                            'This model usually expects 1 or 3 input channels. '
                            'However, it was passed an input_shape with ' +
                            str(input_shape[-1]) + ' input channels.')
                    default_shape = (default_size, default_size, input_shape[-1])
            else:
                if data_format == 'channels_first':
                    default_shape = (3, default_size, default_size)
                else:
                    default_shape = (default_size, default_size, 3)
            if weights == 'imagenet' and require_flatten:
                if input_shape is not None:
                    if input_shape != default_shape:
                        raise ValueError('When setting`include_top=True` '
                                         'and loading `imagenet` weights, '
                                         '`input_shape` should be ' +
                                         str(default_shape) + '.')
                return default_shape
            if input_shape:
                if data_format == 'channels_first':
                    if input_shape is not None:
                        if len(input_shape) != 3:
                            raise ValueError(
                                '`input_shape` must be a tuple of three integers.')
                        if input_shape[0] != 3 and weights == 'imagenet':
                            raise ValueError('The input must have 3 channels; got '
                                             '`input_shape=' + str(input_shape) + '`')
                        if ((input_shape[1] is not None and input_shape[1] < min_size) or
                           (input_shape[2] is not None and input_shape[2] < min_size)):
                            raise ValueError('Input size must be at least ' +
                                             str(min_size) + 'x' + str(min_size) + '; got '
                                             '`input_shape=' + str(input_shape) + '`')
                else:
                    if input_shape is not None:
                        if len(input_shape) != 3:
                            raise ValueError(
                                '`input_shape` must be a tuple of three integers.')
                        if input_shape[-1] != 3 and weights == 'imagenet':
                            raise ValueError('The input must have 3 channels; got '
                                             '`input_shape=' + str(input_shape) + '`')
                        if ((input_shape[0] is not None and input_shape[0] < min_size) or
                           (input_shape[1] is not None and input_shape[1] < min_size)):
                            raise ValueError('Input size must be at least ' +
                                             str(min_size) + 'x' + str(min_size) + '; got '
                                             '`input_shape=' + str(input_shape) + '`')
            else:
                if require_flatten:
                    input_shape = default_shape
                else:
                    if data_format == 'channels_first':
                        input_shape = (3, None, None)
                    else:
                        input_shape = (None, None, 3)
            if require_flatten:
                if None in input_shape:
                    raise ValueError('If `include_top` is True, '
                                     'you should specify a static `input_shape`. '
                                     'Got `input_shape=' + str(input_shape) + '`')
            return input_shape


        class Scale(Layer):

            def __init__(
                self,
                weights=None,
                axis=-1,
                momentum=0.9,
                beta_init="zero",
                gamma_init="one",
                **kwargs
            ):
                self.momentum = momentum
                self.axis = axis
                self.beta_init = initializers.get(beta_init)
                self.gamma_init = initializers.get(gamma_init)
                self.initial_weights = weights
                super(Scale, self).__init__(**kwargs)

            def build(self, input_shape):
                self.input_spec = [InputSpec(shape=input_shape)]
                shape = (int(input_shape[self.axis]),)

                self.gamma = K.variable(self.gamma_init(shape), name="%s_gamma" % self.name)
                self.beta = K.variable(self.beta_init(shape), name="%s_beta" % self.name)
                self.trainable_weights = [self.gamma, self.beta]

                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights

            def call(self, x, mask=None):
                input_shape = self.input_spec[0].shape
                broadcast_shape = [1] * len(input_shape)
                broadcast_shape[self.axis] = input_shape[self.axis]

                out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(
                    self.beta, broadcast_shape
                )
                return out

            def get_config(self):
                config = {"momentum": self.momentum, "axis": self.axis}
                base_config = super(Scale, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))


        def identity_block(input_tensor, kernel_size, filters, stage, block):
            eps = 1.1e-5

            if K.common.image_dim_ordering() == "tf":
                bn_axis = 3
            else:
                bn_axis = 1

            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"
            scale_name_base = "scale" + str(stage) + block + "_branch"

            x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + "2a", use_bias=False)(
                input_tensor
            )
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2a")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2a")(x)
            x = Activation("relu", name=conv_name_base + "2a_relu")(x)

            x = ZeroPadding2D((1, 1), name=conv_name_base + "2b_zeropadding")(x)
            x = Conv2D(
                nb_filter2,
                (kernel_size, kernel_size),
                name=conv_name_base + "2b",
                use_bias=False,
            )(x)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2b")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2b")(x)
            x = Activation("relu", name=conv_name_base + "2b_relu")(x)

            x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=False)(x)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2c")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2c")(x)

            x = add([x, input_tensor], name="res" + str(stage) + block)
            x = Activation("relu", name="res" + str(stage) + block + "_relu")(x)
            return x


        def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
            eps = 1.1e-5

            if K.common.image_dim_ordering() == "tf":
                bn_axis = 3
            else:
                bn_axis = 1

            nb_filter1, nb_filter2, nb_filter3 = filters
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"
            scale_name_base = "scale" + str(stage) + block + "_branch"

            x = Conv2D(
                nb_filter1, (1, 1), strides=strides, name=conv_name_base + "2a", use_bias=False
            )(input_tensor)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2a")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2a")(x)
            x = Activation("relu", name=conv_name_base + "2a_relu")(x)

            x = ZeroPadding2D((1, 1), name=conv_name_base + "2b_zeropadding")(x)
            x = Conv2D(
                nb_filter2,
                (kernel_size, kernel_size),
                name=conv_name_base + "2b",
                use_bias=False,
            )(x)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2b")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2b")(x)
            x = Activation("relu", name=conv_name_base + "2b_relu")(x)

            x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", use_bias=False)(x)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "2c")(x)
            x = Scale(axis=bn_axis, name=scale_name_base + "2c")(x)

            shortcut = Conv2D(
                nb_filter3, (1, 1), strides=strides, name=conv_name_base + "1", use_bias=False
            )(input_tensor)
            shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + "1")(
                shortcut
            )
            shortcut = Scale(axis=bn_axis, name=scale_name_base + "1")(shortcut)

            x = add([x, shortcut], name="res" + str(stage) + block)
            x = Activation("relu", name="res" + str(stage) + block + "_relu")(x)
            return x


        def ResNet152(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            large_input=False,
            pooling=None,
            classes=1000,
        ):
            if weights not in {"imagenet", None}:
                raise ValueError(
                    "The `weights` argument should be either "
                    "`None` (random initialization) or `imagenet` "
                    "(pre-training on ImageNet)."
                )

            if weights == "imagenet" and include_top and classes != 1000:
                raise ValueError(
                    "If using `weights` as imagenet with `include_top`"
                    " as true, `classes` should be 1000"
                )

            eps = 1.1e-5

            if large_input:
                img_size = 448
            else:
                img_size = 224

            # Determine proper input shape
            input_shape = _obtain_input_shape(
                input_shape,
                default_size=img_size,
                min_size=197,
                data_format=K.image_data_format(),
                require_flatten=include_top,
            )

            if input_tensor is None:
                img_input = Input(shape=input_shape)
            else:
                if not K.is_keras_tensor(input_tensor):
                    img_input = Input(tensor=input_tensor, shape=input_shape)
                else:
                    img_input = input_tensor

            # handle dimension ordering for different backends
            if K.common.image_dim_ordering() == "tf":
                bn_axis = 3
            else:
                bn_axis = 1

            x = ZeroPadding2D((3, 3), name="conv1_zeropadding")(img_input)
            x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", use_bias=False)(x)
            x = BatchNormalization(epsilon=eps, axis=bn_axis, name="bn_conv1")(x)
            x = Scale(axis=bn_axis, name="scale_conv1")(x)
            x = Activation("relu", name="conv1_relu")(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

            x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
            x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
            x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

            x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
            for i in range(1, 8):
                x = identity_block(x, 3, [128, 128, 512], stage=3, block="b" + str(i))

            x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
            for i in range(1, 36):
                x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b" + str(i))

            x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

            if large_input:
                x = AveragePooling2D((14, 14), name="avg_pool")(x)
            else:
                x = AveragePooling2D((7, 7), name="avg_pool")(x)

            # include classification layer by default, not included for feature extraction
            if include_top:
                x = Flatten()(x)
                x = Dense(classes, activation="softmax", name="fc1000")(x)
            else:
                if pooling == "avg":
                    x = GlobalAveragePooling2D()(x)
                elif pooling == "max":
                    x = GlobalMaxPooling2D()(x)

            # Ensure that the model takes into account
            # any potential predecessors of `input_tensor`.
            if input_tensor is not None:
                inputs = get_source_inputs(input_tensor)
            else:
                inputs = img_input
            # Create model.
            model = Model(inputs, x, name="resnet152")

            # load weights
            if weights == "imagenet":
                if include_top:
                    weights_path = get_file(
                        "resnet152_weights_tf.h5",
                        WEIGHTS_PATH,
                        cache_subdir="models",
                        md5_hash="cdb18a2158b88e392c0905d47dcef965",
                    )
                else:
                    weights_path = get_file(
                        "resnet152_weights_tf_notop.h5",
                        WEIGHTS_PATH_NO_TOP,
                        cache_subdir="models",
                        md5_hash="4a90dcdafacbd17d772af1fb44fc2660",
                    )
                model.load_weights(weights_path, by_name=True)
                if K.backend() == "theano":
                    layer_utils.convert_all_kernels_in_model(model)
                    if include_top:
                        maxpool = model.get_layer(name="avg_pool")
                        shape = maxpool.output_shape[1:]
                        dense = model.get_layer(name="fc1000")
                        layer_utils.convert_dense_weights_data_format(
                            dense, shape, "channels_first"
                        )

                if K.image_data_format() == "channels_first" and K.backend() == "tensorflow":
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
            return model


        if __name__ == "__main__":
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import tensorflow as tf

                tf.logging.set_verbosity(tf.logging.ERROR)
                import os
                os.makedirs("outputs", exist_ok=True)

                model = ResNet152(include_top=False, input_shape=(200, 200, 3), pooling="avg", weights="imagenet")
                model.save_weights("outputs/model.pkl")

        """


class MockWorkspaceCreationTests:
    """Workspace Creation Test Suite"""

    @pytest.fixture(scope="class")
    def context_type(self):
        """
        Abstract Workspace Type Fixture - Update with Workspace Context to test
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        """

        :return:
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def realtime_score_context(
        self, context_type: RealtimeScoreAKSContext, files_for_testing
    ) -> RealtimeScoreAKSContext:
        """
        Get or Create Context for Testing
        :param context_type: impl of WorkspaceContext
        :param test_files: Dict of input Files
        :return:
        """
        raise NotImplementedError

    def test_mock_get_or_create(
        self,
        realtime_score_context: RealtimeScoreAKSContext,
        context_type: WorkspaceContext,
    ):
        """
        Assert Context Type and Creation

        :param realtime_score_context: Testing Context
        :param context_type: Expected Context Type
        """
        assert realtime_score_context
        assert hasattr(realtime_score_context, "_subscription_id")
        assert hasattr(realtime_score_context, "_resource_group")
        assert hasattr(realtime_score_context, "_workspace_name")
        assert hasattr(realtime_score_context, "project_configuration_file")
        assert hasattr(realtime_score_context, "score_py")
        assert hasattr(realtime_score_context, "train_py")

    def test_mock_get_or_create_model(
        self, monkeypatch, realtime_score_context: MLRealtimeScore
    ):
        """

        :param realtime_score_context: Testing Context
        """

        @staticmethod
        def mockreturn_2(
            workspace, name, id, tags, properties, version, model_framework, run_id
        ):
            return {
                "name": "mock",
                "id": "1",
                "createdTime": "11/8/2020",
                "description": "",
                "mimeType": "a",
                "properties": "",
                "unpack": "",
                "url": "localhost",
                "version": 1,
                "experimentName": "expName",
                "runId": 1,
                "datasets": None,
                "createdBy": "mock",
                "framework": "python",
                "frameworkVersion": "1",
            }

        def mock_get_model_path_remote(model_name, version, workspace):
            return "."

        def mock_initialize(self, workspace, obj_dict):
            pass

        monkeypatch.setattr(Model, "_get", mockreturn_2)
        monkeypatch.setattr(Model, "_get_model_path_remote", mock_get_model_path_remote)
        monkeypatch.setattr(Model, "_initialize", mock_initialize)
        realtime_score_context.prepare_data(".")
        assert realtime_score_context.get_or_create_model()

        assert os.path.isfile("model.pkl")

    def test_mock_get_compute_targets(
        self, realtime_score_context: RealtimeScoreAKSContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.compute_targets

    def test_mock_get_webservices(
        self, realtime_score_context: RealtimeScoreAKSContext
    ):
        """

        :param realtime_score_context: Testing Context
        """
        assert realtime_score_context.webservices

    @property
    def models(self):
        """Return a dictionary where the key is model name, and value is a :class:`azureml.core.model.Model` object.

        Raises a :class:`azureml.exceptions.WebserviceException` if there was a problem interacting with
        model management service.

        :return: A dictionary of models.
        :rtype: dict[str, azureml.core.Model]
        :raises: azureml.exceptions.WebserviceException
        """
        return {}


class TestMockDeployRTS(MockWorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return MLRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_model.py", "score_py": "driver.py"}

    @pytest.fixture
    def realtime_score_context(
        self, monkeypatch, context_type: MLRealtimeScore, files_for_testing
    ) -> MLRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """

        def mockreturn(train_py, score_py):
            project_configuration = ProjectConfiguration(project_configuration_file)
            assert project_configuration.has_value("subscription_id")
            assert project_configuration.has_value("resource_group")
            assert project_configuration.has_value("workspace_name")
            ws = MockMLRealtimeScore(
                subscription_id=project_configuration.get_value("subscription_id"),
                resource_group=project_configuration.get_value("resource_group"),
                workspace_name=project_configuration.get_value("workspace_name"),
                configuration_file=project_configuration_file,
                score_py=score_py,
                train_py=train_py,
            )
            return ws

        monkeypatch.setattr(context_type, "get_or_create_workspace", mockreturn)

        return context_type.get_or_create_workspace(
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )


class TestMockDeployDeepRTS(MockWorkspaceCreationTests):
    @pytest.fixture(scope="class")
    def context_type(self):
        """

        :return:
        """
        return DeepRealtimeScore

    @pytest.fixture(scope="class")
    def files_for_testing(self):
        return {"train_py": "create_deep_model.py", "score_py": "deep_driver.py"}

    @pytest.fixture
    def realtime_score_context(
        self, monkeypatch, context_type: MLRealtimeScore, files_for_testing
    ) -> DeepRealtimeScore:
        """
        Get or Create Context for Testing
        :param files_for_testing:
        :param context_type: impl of WorkspaceContext
        :return:
        """

        def mockreturn(train_py, score_py):
            project_configuration = ProjectConfiguration(project_configuration_file)
            assert project_configuration.has_value("subscription_id")
            assert project_configuration.has_value("resource_group")
            assert project_configuration.has_value("workspace_name")
            ws = MockDeepRealtimeScore(
                project_configuration.get_value("subscription_id"),
                project_configuration.get_value("resource_group"),
                project_configuration.get_value("workspace_name"),
                project_configuration_file,
                score_py=score_py,
                train_py=train_py,
            )
            return ws

        monkeypatch.setattr(context_type, "get_or_create_workspace", mockreturn)

        return context_type.get_or_create_workspace(
            train_py=files_for_testing["train_py"],
            score_py=files_for_testing["score_py"],
        )

    def test_mock_get_or_create_model(
        self, monkeypatch, realtime_score_context: DeepRealtimeScore
    ):
        """

        :param realtime_score_context: Testing Context
        """

        if not os.path.isfile("script/train.py"):
            os.makedirs("script", exist_ok=True)

            create_model_py = DEEP_TRAIN_PY
            with open("script/train.py", "w") as file:
                file.write(create_model_py)

        assert os.path.isfile("script/train.py")
        @staticmethod
        def mockreturn_2(
            workspace, name, id, tags, properties, version, model_framework, run_id
        ):
            return {
                "name": "mock",
                "id": "1",
                "createdTime": "11/8/2020",
                "description": "",
                "mimeType": "a",
                "properties": "",
                "unpack": "",
                "url": "localhost",
                "version": 1,
                "experimentName": "expName",
                "runId": 1,
                "datasets": None,
                "createdBy": "mock",
                "framework": "python",
                "frameworkVersion": "1",
            }

        def mock_get_model_path_remote(model_name, version, workspace):
            return "."

        def mock_initialize(self, workspace, obj_dict):
            pass

        monkeypatch.setattr(Model, "_get", mockreturn_2)
        monkeypatch.setattr(Model, "_get_model_path_remote", mock_get_model_path_remote)
        monkeypatch.setattr(Model, "_initialize", mock_initialize)
        assert realtime_score_context.get_or_create_model()

        assert os.path.isfile("outputs/model.pkl")
