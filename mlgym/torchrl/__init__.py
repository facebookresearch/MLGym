# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchrl
import tensordict

assert not torchrl.auto_unwrap_transformed_env(), "Make sure torchrl set_auto_unwrap_transformed_env() or AUTO_UNWRAP_TRANSFORMED_ENV is set to False"
assert not tensordict.capture_non_tensor_stack(), "Make sure tensordict set_capture_non_tensor_stack() or CAPTURE_NONTENSOR_STACK is set to False"

from .transforms import make_env, IsolateCodeBlock, ResetModule, ReadState, StateToMessage, MessageToHistory, TemplateTransform
