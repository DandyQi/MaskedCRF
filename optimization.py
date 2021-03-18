# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops


def create_optimizer(loss, init_lr, lr_decay_factor=1.0):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    tf.summary.scalar("learning_rate", learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)

    optimizer = MutilGPUAdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )
    tvars = tf.trainable_variables()

    if lr_decay_factor < 1.0:
        tvars = [tvar for tvar in tvars if tvar.name.startswith != "bert/embeddings"]

    gvs = optimizer.compute_gradients(loss, tvars)
    gvs = [(g, v) for g, v in gvs if g is not None]
    grads, tvars = list(zip(*gvs))

    if lr_decay_factor < 1.0:
        layer_wise_decay_map = {}
        for i in range(12):
            layer_wise_decay_map["bert/encoder/layer_%s/" % i] = 11 - i

        layer_wise_decay_grads = []
        for grad, var in zip(grads, tvars):
            for prefix, decay_power in layer_wise_decay_map.items():
                if var.name.startswith(prefix):
                    decay_lr = init_lr * pow(lr_decay_factor, decay_power)
                    grad = grad * pow(lr_decay_factor, decay_power)
                    tf.logging.info("%s using lr: %s" % (var.name, decay_lr))
                    break
            layer_wise_decay_grads.append(grad)

        grads = layer_wise_decay_grads

    all_finite = tf.constant(True, dtype=tf.bool)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0,
                                        use_norm=tf.cond(
                                            all_finite,
                                            lambda: tf.global_norm(grads),
                                            lambda: tf.constant(1.0)))

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = tf.cond(all_finite, lambda: global_step + 1, lambda: global_step)
    new_global_step = tf.identity(new_global_step, name='update_step')
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class MutilGPUAdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(MutilGPUAdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.identity(learning_rate, name='learning_rate')
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare(self):
        self.learning_rate_t = ops.convert_to_tensor(
            self.learning_rate, name='learning_rate')
        self.weight_decay_rate_t = ops.convert_to_tensor(
            self.weight_decay_rate, name='weight_decay_rate')
        self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
        self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
        self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'm', self._name)
            self._zeros_slot(v, 'v', self._name)

    def _apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = state_ops.assign(m, m * beta_1_t,
                               use_locking=self._use_locking)

        m_scaled_g_values = grad * (1 - beta_1_t)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        var_update = state_ops.assign_sub(var,
                                          update_with_lr,
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(
                    x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True
