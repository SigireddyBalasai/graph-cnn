
�jroot"_tf_keras_network*�j{"name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model_9", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_222", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_222", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_22", "inbound_nodes": [[["conv2d_222", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_119", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_119", "inbound_nodes": [[["max_pooling2d_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_234", "trainable": true, "dtype": "float32", "rate": 0.6970167077069287, "noise_shape": null, "seed": null}, "name": "dropout_234", "inbound_nodes": [[["activation_119", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_23", "inbound_nodes": [[["dropout_234", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_24", "inbound_nodes": [[["dropout_234", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_121", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_121", "inbound_nodes": [[["max_pooling2d_23", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_123", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_123", "inbound_nodes": [[["max_pooling2d_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_110", "inbound_nodes": [[["activation_121", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_112", "inbound_nodes": [[["activation_123", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_237", "trainable": true, "dtype": "float32", "rate": 0.3808116740560216, "noise_shape": null, "seed": null}, "name": "dropout_237", "inbound_nodes": [[["batch_normalization_110", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_240", "trainable": true, "dtype": "float32", "rate": 0.34013342336149877, "noise_shape": null, "seed": null}, "name": "dropout_240", "inbound_nodes": [[["batch_normalization_112", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_28", "trainable": true, "dtype": "float32"}, "name": "add_28", "inbound_nodes": [[["dropout_237", 0, 0, {}], ["dropout_240", 0, 0, {}]]]}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_31", "inbound_nodes": [[["add_28", 0, 0, {}]]]}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_32", "inbound_nodes": [[["dropout_237", 0, 0, {}]]]}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_33", "inbound_nodes": [[["dropout_240", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["aux_layer_31", 0, 0], ["aux_layer_32", 0, 0], ["aux_layer_33", 0, 0]]}, "shared_object_id": 27, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_10"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_10"]}, "keras_version": "2.15.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_222", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_222", "inbound_nodes": [[["input_10", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_22", "inbound_nodes": [[["conv2d_222", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Activation", "config": {"name": "activation_119", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_119", "inbound_nodes": [[["max_pooling2d_22", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dropout", "config": {"name": "dropout_234", "trainable": true, "dtype": "float32", "rate": 0.6970167077069287, "noise_shape": null, "seed": null}, "name": "dropout_234", "inbound_nodes": [[["activation_119", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_23", "inbound_nodes": [[["dropout_234", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_24", "inbound_nodes": [[["dropout_234", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Activation", "config": {"name": "activation_121", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_121", "inbound_nodes": [[["max_pooling2d_23", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Activation", "config": {"name": "activation_123", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_123", "inbound_nodes": [[["max_pooling2d_24", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_110", "inbound_nodes": [[["activation_121", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_112", "inbound_nodes": [[["activation_123", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Dropout", "config": {"name": "dropout_237", "trainable": true, "dtype": "float32", "rate": 0.3808116740560216, "noise_shape": null, "seed": null}, "name": "dropout_237", "inbound_nodes": [[["batch_normalization_110", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_240", "trainable": true, "dtype": "float32", "rate": 0.34013342336149877, "noise_shape": null, "seed": null}, "name": "dropout_240", "inbound_nodes": [[["batch_normalization_112", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Add", "config": {"name": "add_28", "trainable": true, "dtype": "float32"}, "name": "add_28", "inbound_nodes": [[["dropout_237", 0, 0, {}], ["dropout_240", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_31", "inbound_nodes": [[["add_28", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_32", "inbound_nodes": [[["dropout_237", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "AuxLayer", "config": {"num_classes": 102}, "name": "aux_layer_33", "inbound_nodes": [[["dropout_240", 0, 0, {}]]], "shared_object_id": 26}], "input_layers": [["input_10", 0, 0]], "output_layers": [["aux_layer_31", 0, 0], ["aux_layer_32", 0, 0], ["aux_layer_33", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "aux_layer_31_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 29}], [{"class_name": "MeanMetricWrapper", "config": {"name": "aux_layer_32_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 30}], [{"class_name": "MeanMetricWrapper", "config": {"name": "aux_layer_33_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 31}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}2
�
root.layer_with_weights-0"_tf_keras_layer*�	{"name": "conv2d_222", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "conv2d_222", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_10", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}2
�root.layer-2"_tf_keras_layer*�{"name": "max_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_222", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}2
�root.layer-3"_tf_keras_layer*�{"name": "activation_119", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_119", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["max_pooling2d_22", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 32]}}2
�root.layer-4"_tf_keras_layer*�{"name": "dropout_234", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_234", "trainable": true, "dtype": "float32", "rate": 0.6970167077069287, "noise_shape": null, "seed": null}, "inbound_nodes": [[["activation_119", 0, 0, {}]]], "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 32]}}2
�root.layer-5"_tf_keras_layer*�{"name": "max_pooling2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["dropout_234", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 32]}}2
�root.layer-6"_tf_keras_layer*�{"name": "max_pooling2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_24", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["dropout_234", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 32]}}2
�root.layer-7"_tf_keras_layer*�{"name": "activation_121", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_121", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["max_pooling2d_23", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�	root.layer-8"_tf_keras_layer*�{"name": "activation_123", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_123", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["max_pooling2d_24", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�

root.layer_with_weights-1"_tf_keras_layer*�	{"name": "batch_normalization_110", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["activation_121", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�
root.layer_with_weights-2"_tf_keras_layer*�	{"name": "batch_normalization_112", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["activation_123", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�
�
�
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "aux_layer_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "AuxLayer", "config": {"num_classes": 102}, "inbound_nodes": [[["add_28", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�root.layer_with_weights-4"_tf_keras_layer*�{"name": "aux_layer_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "AuxLayer", "config": {"num_classes": 102}, "inbound_nodes": [[["dropout_237", 0, 0, {}]]], "shared_object_id": 25, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
�root.layer_with_weights-5"_tf_keras_layer*�{"name": "aux_layer_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "AuxLayer", "config": {"num_classes": 102}, "inbound_nodes": [[["dropout_240", 0, 0, {}]]], "shared_object_id": 26, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
��'root.layer_with_weights-3.layers_list.0"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
��'root.layer_with_weights-3.layers_list.1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 40, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��'root.layer_with_weights-3.layers_list.2"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��'root.layer_with_weights-4.layers_list.0"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
��'root.layer_with_weights-4.layers_list.1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 47, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��'root.layer_with_weights-4.layers_list.2"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��'root.layer_with_weights-5.layers_list.0"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 32]}}2
��'root.layer_with_weights-5.layers_list.1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 54, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��'root.layer_with_weights-5.layers_list.2"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 102, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 59}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "Mean", "name": "aux_layer_31_loss", "dtype": "float32", "config": {"name": "aux_layer_31_loss", "dtype": "float32"}, "shared_object_id": 60}2
��root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "Mean", "name": "aux_layer_32_loss", "dtype": "float32", "config": {"name": "aux_layer_32_loss", "dtype": "float32"}, "shared_object_id": 61}2
��root.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "Mean", "name": "aux_layer_33_loss", "dtype": "float32", "config": {"name": "aux_layer_33_loss", "dtype": "float32"}, "shared_object_id": 62}2
��root.keras_api.metrics.4"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "aux_layer_31_accuracy", "dtype": "float32", "config": {"name": "aux_layer_31_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 29}2
��root.keras_api.metrics.5"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "aux_layer_32_accuracy", "dtype": "float32", "config": {"name": "aux_layer_32_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 30}2
��root.keras_api.metrics.6"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "aux_layer_33_accuracy", "dtype": "float32", "config": {"name": "aux_layer_33_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 31}2