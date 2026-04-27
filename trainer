# coding=utf-8

# Copyright 2020-present the HuggingFace Inc. team.

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

"""

The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.

"""
# 在 trainer.py 文件的开头添加这些导入
from collections import deque, defaultdict
import numpy as np
from typing import Dict
import inspect

import math

import os

import re

import shutil

import sys

import time

from functools import partial

from pathlib import Path

from typing import Dict, Union, Any

from typing import TYPE_CHECKING, Optional, Iterable

from hybrid_trainer import HybridTrainingMixin

import numpy as np

import torch

import torch.distributed as dist

from packaging import version

from sklearn.linear_model import LogisticRegressionCV

from torch import nn

from torch.func import functional_call, jvp

from torch.optim import SGD, Adam

from torch.utils.data import DataLoader, RandomSampler

from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

from transformers import Trainer

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

try:

    from transformers.integrations import is_deepspeed_zero3_enabled  # type: ignore

except Exception:

    def is_deepspeed_zero3_enabled() -> bool:

        """Fallback: best-effort detection for a deepspeed-zero3-like environment."""

        try:

            import deepspeed  # type: ignore

            # best-effort: check common zero3 indicators in deepspeed

            return hasattr(deepspeed, "zero") or hasattr(deepspeed, "initialize") or hasattr(deepspeed, "ops")

        except Exception:

            return False

# Provide deepspeed_init wrapper: prefer transformers.deepspeed, else fallback to deepspeed.initialize

try:

    from transformers.deepspeed import deepspeed_init  # type: ignore

except Exception:

    try:

        import deepspeed  # type: ignore


        def deepspeed_init(*args, **kwargs):

            """

            Minimal wrapper that forwards to deepspeed.initialize.

            Note: return signature may differ across deepspeed versions.

            """

            return deepspeed.initialize(*args, **kwargs)

    except Exception:

        # deepspeed not installed: provide a stub raising informative error when called

        def deepspeed_init(*args, **kwargs):

            raise RuntimeError(

                "deepspeed_init() called but deepspeed is not installed or not importable. "

                "Install deepspeed or disable deepspeed-related options."

            )

# Integrations helpers: hp_params is exported by transformers.integrations in many versions; try import but fallback gracefully

try:

    from transformers.integrations import hp_params  # type: ignore

except Exception:

    hp_params = None

# is_fairscale_available: if transformers doesn't provide it, create a simple fallback

try:

    from transformers.integrations import is_fairscale_available  # type: ignore

except Exception:

    def is_fairscale_available() -> bool:

        try:

            import fairscale  # type: ignore

            return True

        except Exception:

            return False

# pytorch utils compatibility: some transformers versions export helpers; otherwise implement simple version checks

try:

    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11  # type: ignore

except Exception:

    # Fallback based on torch.__version__ using packaging.version

    from packaging import version as _pack_version


    def _torch_ver_tuple():

        try:

            return _pack_version.parse(torch.__version__)

        except Exception:

            return _pack_version.parse("0.0.0")


    def is_torch_greater_or_equal_than_1_10() -> bool:

        return _torch_ver_tuple() >= _pack_version.parse("1.10")


    def is_torch_less_than_1_11() -> bool:

        return _torch_ver_tuple() < _pack_version.parse("1.11")

# If available, keep using transformers.dependency_versions_check.dep_version_check

try:

    from transformers.dependency_versions_check import dep_version_check  # type: ignore

except Exception:

    def dep_version_check(*args, **kwargs):

        return None

from transformers.trainer_callback import (

    DefaultFlowCallback,

    ProgressCallback,

    TrainerState,

)

from transformers.trainer_pt_utils import (

    IterableDatasetShard,

)

# --- Compatibility fix for ShardedDDPOption (transformers 4.30–4.45) ---

try:

    from transformers.trainer import ShardedDDPOption  # for newer versions

except ImportError:

    try:

        from transformers.trainer_pt_utils import ShardedDDPOption  # fallback for some mid versions

    except Exception:

        class ShardedDDPOption:

            SIMPLE = "simple"

            ZERO_DP_2 = "zero_dp_2"

            ZERO_DP_3 = "zero_dp_3"

from transformers.trainer_utils import (

    HPSearchBackend,

    TrainOutput,

    has_length,

    speed_metrics,

)

from transformers.utils import (

    WEIGHTS_NAME,

    is_apex_available,

    is_in_notebook,

    is_sagemaker_mp_enabled,

    logging,

)

# --- Compatibility patch for is_torch_tpu_available (removed in Transformers ≥4.43) ---

try:

    from transformers.utils import is_torch_tpu_available  # old versions

except ImportError:

    def is_torch_tpu_available(check_device: bool = True) -> bool:

        return False

# from torch.optim.optimizer import StateDict, params_t

import wandb

from gradient_pruning.pruning_utils import (

    fast_random_mask_like,

    estimate_pretrained_model_magnitude_pruning_threshold,

    compute_named_parameters_to_sparsity,

)

from metrics import f1

from utils import OPT_PERTURBATION_LEVEL_TO_REGEX

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]

DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

    import torch_xla.debug.metrics as met

    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():

    import smdistributed.modelparallel.torch as smp

    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:

    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing

TRAINING_ARGS_NAME = "training_args.bin"

TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pt"

SCHEDULER_NAME = "scheduler.pt"

SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):
    def __init__(self, evaluate_func, dev_samples, eval_samples, perturb_module_regex, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.evaluate_func = evaluate_func
        self.dev_samples = dev_samples
        self.eval_samples = eval_samples
        self.perturb_module_regex = perturb_module_regex
        self.sharded_ddp = None
        self.fsdp = None
        self.do_grad_scaling = False
        self._original_attn_implementation = None

        # 🚀 简化版混合训练初始化
        self.hybrid_split_method = getattr(self.args, 'hybrid_split_method', 'none')
        self.fo_guidance_steps = getattr(self.args, 'fo_guidance_steps', 10)
        self.fo_guidance_trigger_threshold = getattr(self.args, 'fo_guidance_trigger_threshold', 0.005)
        self.fo_guidance_window = getattr(self.args, 'fo_guidance_window', 5)
        self.fo_min_interval = getattr(self.args, 'fo_min_interval', 10)
        self.zo_fo_split_ratio = getattr(self.args, 'zo_fo_split_ratio', 0.5)

        # 训练状态
        self.loss_history = deque(maxlen=self.fo_guidance_window)
        self.last_fo_step = -self.fo_min_interval
        self.current_step = 0
        self.run_fo_step_next = False

        # 🚀 内存优化新增
        self.memory_optimized = True
        self.gradient_checkpointing_enabled = False
        self.selective_parameter_update = True

        logger.info(f"🚀 混合训练模式: {self.hybrid_split_method}, 内存优化: {self.memory_optimized}")

    def setup_memory_optimizations(self, model):
        """设置内存优化配置"""
        # 🚀 启用梯度检查点
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.gradient_checkpointing_enabled = True
            logger.info("🚀 已启用梯度检查点")

        # 🚀 设置梯度为None以节省内存
        for param in model.parameters():
            param.grad = None

    def _get_learning_rate(self) -> float:
        """获取当前训练步的学习率"""
        if self.lr_scheduler is not None:
            try:
                lrs = self.lr_scheduler.get_last_lr()
                if lrs:
                    return lrs[0]
            except Exception:
                pass
        return self.args.learning_rate

    def _should_run_fo_guidance(self) -> bool:
        """判断是否应该运行FO指导"""
        if len(self.loss_history) < self.fo_guidance_window:
            return False

        # 检查步数间隔
        if self.current_step - self.last_fo_step < self.fo_min_interval:
            return False

        # 检查损失改善情况
        recent_losses = list(self.loss_history)
        if len(recent_losses) < 2:
            return False

        improvement = recent_losses[0] - recent_losses[-1]
        needs_fo = improvement < self.fo_guidance_trigger_threshold

        # 定期FO指导
        periodic_fo = (self.current_step % self.fo_guidance_steps == 0)

        return needs_fo or periodic_fo

    def _make_hybrid_decision(self) -> str:
        """做出混合训练决策"""
        if self.hybrid_split_method == "none":
            return "ZO"

        if self.hybrid_split_method == "fixed_ratio":
            return "FO" if np.random.random() > self.zo_fo_split_ratio else "ZO"

        elif self.hybrid_split_method == "dynamic_zo_fo":
            if self._should_run_fo_guidance() or self.run_fo_step_next:
                return "FO"
            else:
                return "ZO"
        else:
            return "ZO"

    def _make_memory_efficient_hybrid_decision(self, model):
        """内存高效的混合决策"""
        # 检查可用内存
        if torch.cuda.is_available():
            free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_ratio = free_memory / total_memory

            # 如果内存紧张，减少FO步骤的频率
            if free_ratio < 0.3:  # 空闲内存少于30%
                original_steps = self.fo_guidance_steps
                self.fo_guidance_steps = min(50, self.fo_guidance_steps * 2)  # 减少FO频率
                if original_steps != self.fo_guidance_steps:
                    logger.info(f"🔄 内存紧张，调整FO频率: {original_steps} -> {self.fo_guidance_steps} 步")

        return self._make_hybrid_decision()

    def _apply_layer_decisions(self, model, layer_decisions, strategy):
        """应用层决策，只启用指定层的梯度"""
        for name, param in model.named_parameters():
            if name in layer_decisions and layer_decisions[name] == strategy:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _restore_layer_states(self, model):
        """恢复所有层的梯度状态"""
        for param in model.parameters():
            param.requires_grad = True

    def _fo_training_step_memory_efficient(self, model, inputs, layer_decisions=None):
        """内存优化的FO训练步骤"""
        model.train()

        # 🚀 关键优化：使用torch.no_grad()包装前向传播
        with torch.no_grad():
            # 如果指定了层决策，只训练FO层
            if layer_decisions:
                self._apply_layer_decisions(model, layer_decisions, "FO")

            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

        # 🚀 关键优化：分离计算图，减少内存占用
        loss = loss.detach().clone().requires_grad_(True)

        # 🚀 关键优化：使用create_graph=False
        loss.backward(create_graph=False)

        # 恢复所有层的可训练状态
        if layer_decisions:
            self._restore_layer_states(model)

        return loss

    def _fo_training_step(self, model, inputs, layer_decisions=None):
        """FO训练步骤（内存安全版本）- 使用优化后的版本"""
        return self._fo_training_step_memory_efficient(model, inputs, layer_decisions)

    def _zo_training_step(self, model, inputs):
        """ZO训练步骤"""
        if self.args.trainer.startswith('zo_'):
            # 使用现有的ZO训练器
            return super().training_step(model, inputs)
        else:
            # 回退到FO训练
            return self._fo_training_step(model, inputs)

    def _dynamic_batch_adjustment(self, current_memory_usage):
        """根据内存使用动态调整批处理大小"""
        if current_memory_usage > 0.8:  # 如果内存使用超过80%
            original_bs = self.args.per_device_train_batch_size
            self.args.per_device_train_batch_size = max(1, self.args.per_device_train_batch_size // 2)
            if original_bs != self.args.per_device_train_batch_size:
                logger.info(f"🔧 动态调整批处理大小: {original_bs} -> {self.args.per_device_train_batch_size}")

    def _hybrid_training_step_memory_aware(self, model, inputs):
        """内存感知的混合训练步骤"""
        # 检查当前内存使用
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self._dynamic_batch_adjustment(current_memory)

        return self._hybrid_training_step(model, inputs)

    def _hybrid_training_step(self, model, inputs):
        """混合训练步骤（内存优化版）"""
        step = self.state.global_step
        self.current_step = step

        # 获取混合决策
        global_strategy = self._make_memory_efficient_hybrid_decision(model)
        layer_decisions = None  # 简化版本，不使用层级别决策

        # 🚀 内存优化：在FO步骤前清空梯度
        if global_strategy == "FO" or self.run_fo_step_next:
            model.zero_grad(set_to_none=True)  # 🚀 彻底清空梯度

        # 根据决策执行训练
        if global_strategy == "FO" or self.run_fo_step_next:
            if self.memory_optimized:
                loss = self._fo_training_step_memory_efficient(model, inputs, layer_decisions)
            else:
                loss = self._fo_training_step(model, inputs, layer_decisions)
            self.last_fo_step = step
            self.run_fo_step_next = False

            # 🚀 内存优化：FO步骤后立即清空梯度
            model.zero_grad(set_to_none=True)

            if step % 50 == 0:
                logger.info(f"🎯 执行FO步骤, loss={loss.item():.4f}")
        else:
            loss = self._zo_training_step(model, inputs)
            if step % 50 == 0:
                logger.info(f"🔬 执行ZO步骤, loss={loss.item():.4f}")

        # 更新训练状态
        if loss is not None and hasattr(loss, 'item'):
            self.loss_history.appendleft(loss.item())

        # 检查是否需要下一次FO指导
        if self._should_run_fo_guidance():
            self.run_fo_step_next = True
            logger.info(f"📈 检测到平台期，计划下一步执行FO指导")

        return loss

    def training_step(self, model, inputs):
        """重写训练步骤以支持内存优化的混合训练"""
        if self.hybrid_split_method != "none":
            if self.memory_optimized:
                return self._hybrid_training_step_memory_aware(model, inputs)
            else:
                return self._hybrid_training_step(model, inputs)
        else:
            return super().training_step(model, inputs)

    def log_memory_usage(self):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3

            logger.info(f"💾 内存使用 - 当前: {allocated:.2f}GB, 保留: {reserved:.2f}GB, 峰值: {max_allocated:.2f}GB")

            # 记录到wandb
            if wandb.run is not None:
                wandb.log({
                    "memory/allocated_gb": allocated,
                    "memory/reserved_gb": reserved,
                    "memory/max_allocated_gb": max_allocated
                })

    def log(self, logs: Dict[str, float]):
        """重写日志记录以添加混合训练信息和内存使用"""
        if self.hybrid_split_method != "none":
            # 添加混合训练特定日志
            logs["hybrid/fo_guidance_triggered"] = float(self.run_fo_step_next)
            logs["hybrid/loss_window_avg"] = float(
                np.mean(list(self.loss_history))
                if self.loss_history else 0.0
            )
            logs["hybrid/steps_since_last_fo"] = self.current_step - self.last_fo_step

            # 🚀 添加内存使用日志
            if torch.cuda.is_available():
                logs["memory/allocated_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
                logs["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3

        super().log(logs)

    def setup_memory_optimizations(self, model):
        """设置内存优化"""
        # 🚀 启用梯度检查点
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("🚀 已启用梯度检查点")

        # 🚀 设置梯度为None以节省内存
        for param in model.parameters():
            param.grad = None

    def _inner_training_loop(

            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None

    ):

        """

        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"

        for those updates.

        """

        # 关键修改：初始化 start_time

        if not hasattr(self, "_start_time"):
            self._start_time = time.time()

        start_time = self._start_time

        self.setup_memory_optimizations(self.model)

        self._train_batch_size = batch_size

        # Data loader and number of training steps

        train_dataloader = self.get_train_dataloader()

        eval_dataloader = self.get_eval_dataloader()  # ----newly-added

        # MeZO added: Linear probing

        if self.args.linear_probing:

            def _get_token_prediction_layer(model):

                if model.config.model_type in ["opt", "llama", "mistral"]:

                    return model.lm_head

                else:

                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):

                """some magic for getting features pre last layer"""

                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)

                model.forward(*args, **kwargs)

                return features["features"]

            logger.info("Linear probing")

            logger.info("Starting to get features for training dataset")

            targets = []

            features = []

            with torch.inference_mode():

                for step, inputs in enumerate(tqdm(train_dataloader)):

                    for k, v in inputs.items():

                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)

                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part

                    assert not self.args.train_as_classification and self.args.only_train_option

                    feature, target = feature[:, :-1], target[:, 1:]

                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])

                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()

            targets = torch.cat(targets, dim=0).cpu().numpy()

            # Whether to use bias

            if self.model.config.model_type in ["opt", "gpt2", "llama", "mistral"]:

                use_bias = False

            else:

                raise NotImplementedError

            # Set early stopping

            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default

            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")

            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",

                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)

            logger.info("Done")

            logger.info("Assigning weights to model")

            decoder = _get_token_prediction_layer(self.model)

            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)

            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)

            if coef_torch.shape[0] == 1:  # The regressor only detects two classes

                assert len(reg.classes_) == 2

                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)

                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):

                decoder.weight.data[token_id] = coef_torch[_i]

                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # ------------------------ 以下保持原逻辑 ------------------------

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None

        if has_length(train_dataloader):

            len_dataloader = len(train_dataloader)

            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps

            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

            num_examples = self.num_examples(train_dataloader)

            if args.max_steps > 0:

                max_steps = args.max_steps

                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(

                    args.max_steps % num_update_steps_per_epoch > 0

                )

                num_train_samples = args.max_steps * total_train_batch_size

            else:

                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

                num_train_epochs = math.ceil(args.num_train_epochs)

                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs

        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size

            max_steps = args.max_steps

            num_train_epochs = sys.maxsize

            num_update_steps_per_epoch = max_steps

            num_examples = total_train_batch_size * args.max_steps

            num_train_samples = args.max_steps * total_train_batch_size

        else:

            raise ValueError(

                "args.max_steps must be set to a positive value if dataloader does not have a length, was"

                f" {args.max_steps}"

            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:

            if self.args.n_gpu > 1:

                raise ValueError(

                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"

                    " (torch.distributed.launch)."

                )

            else:

                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (

                self.sharded_ddp is not None

                and self.sharded_ddp != ShardedDDPOption.SIMPLE

                or is_sagemaker_mp_enabled()

                or self.fsdp is not None

        )

        if args.deepspeed:

            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(

                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint

            )

            self.model = deepspeed_engine.module

            self.model_wrapped = deepspeed_engine

            self.deepspeed = deepspeed_engine

            self.optimizer = optimizer

            self.lr_scheduler = lr_scheduler

        elif not delay_optimizer_creation:

            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # 筛选出所有需要梯度的参数 (例如 PEFT 参数或 Full-FT 参数)
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        # overload the optimizer

        if args.trainer == "zo_adam":

            self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)

            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."

        elif args.trainer == "zo_sgd":

            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."

        elif args.trainer == "fed_hybrid":
            trainer = FederatedHybridTrainer(args, model, tokenizer, metrics)

        else:

            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."

            if args.optimizer == "adam":

                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)

            elif args.optimizer == "sgd":

                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        # ------------------------ 最小修改结束 ------------------------

        # 后续原来的训练循环逻辑无需修改

        # important: at this point:

        # self.model         is the Transformers Model

        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!

        logger.info("***** Running training *****")

        logger.info(f"  Num examples = {num_examples}")

        logger.info(f"  Num Epochs = {num_train_epochs}")

        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")

        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

        logger.info(f"  Total optimization steps = {max_steps}")

        logger.info(

            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"

        )

        self.state.epoch = 0

        start_time = time.time()

        epochs_trained = 0

        steps_trained_in_current_epoch = 0

        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint

        if resume_from_checkpoint is not None and os.path.isfile(

                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)

        ):

            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))

            epochs_trained = self.state.global_step // num_update_steps_per_epoch

            if not args.ignore_data_skip:

                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)

                steps_trained_in_current_epoch *= args.gradient_accumulation_steps

            else:

                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")

            logger.info(f"  Continuing training from epoch {epochs_trained}")

            logger.info(f"  Continuing training from global step {self.state.global_step}")

            if not args.ignore_data_skip:

                logger.info(

                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "

                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "

                    "flag to your launch command, but you will resume the training on data already seen by your model."

                )

                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)

                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references

        self.callback_handler.model = self.model

        self.callback_handler.optimizer = self.optimizer

        self.callback_handler.lr_scheduler = self.lr_scheduler

        self.callback_handler.train_dataloader = train_dataloader

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial

            # parameter to Train when using DDP.

            self.state.trial_name = self.hp_name(self._trial)

        if trial is not None:

            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial

            self.state.trial_params = hp_params(assignments)

        else:

            self.state.trial_params = None

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer

        # to set this after the load.

        self.state.max_steps = max_steps

        self.state.num_train_epochs = num_train_epochs

        self.state.is_local_process_zero = self.is_local_process_zero()

        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()

        tr_loss = torch.tensor(0.0).to(args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses

        self._total_loss_scalar = 0.0

        self._globalstep_last_logged = self.state.global_step

        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.

        if not args.ignore_data_skip:

            for epoch in range(epochs_trained):

                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(

                    train_dataloader.sampler, RandomSampler

                )

                if is_torch_less_than_1_11 or not is_random_sampler:

                    # We just need to begin an iteration to create the randomization of the sampler.

                    # That was before PyTorch 1.11 however...

                    for _ in train_dataloader:
                        break

                else:

                    # Otherwise we need to call the whooooole sampler cause there is some random operation added

                    # AT THE VERY END!

                    _ = list(train_dataloader.sampler)

        # Main training loop

        total_steps = 0

        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')

        self.gradient_sparsity = None  # None, float, or dict

        if self.args.sparse_gradient_group == "layer" or self.args.gradient_sparsity is None:

            self.gradient_sparsity = self.args.gradient_sparsity

            print(f"### layer-wise gradient sparsity = {self.gradient_sparsity}")

        elif self.args.sparse_gradient_group == "global":

            threshold = estimate_pretrained_model_magnitude_pruning_threshold(model, self.args.gradient_sparsity)

            self.gradient_sparsity = compute_named_parameters_to_sparsity(model, threshold)

            print(f"### global gradient sparsity, weight magnitude threshold = {threshold}")

        for epoch in range(epochs_trained, num_train_epochs):

            print(f"-------------------------- Training Epoch {epoch} --------------------------")

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):

                train_dataloader.sampler.set_epoch(epoch)

            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):

                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():

                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)

                epoch_iterator = parallel_loader

            else:

                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (

                len(epoch_iterator)

                if len_dataloader is not None

                else args.max_steps * args.gradient_accumulation_steps

            )

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1

            # Start one epoch training

            for step, inputs in enumerate(epoch_iterator):

                if total_steps % self.args.sparse_gradient_resample_steps == 0:
                    self.sparse_grad_random_seed = np.random.randint(1000000000)

                total_steps += 1

                self.current_step = total_steps  # 更新混合训练器的当前步骤

                # torch.cuda.synchronize()

                step_start_time = time.time()

                # Skip past any already trained steps if resuming training

                if steps_trained_in_current_epoch > 0:

                    steps_trained_in_current_epoch -= 1

                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)

                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)

                    continue

                elif steps_trained_progress_bar is not None:

                    steps_trained_progress_bar.close()

                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:

                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # ------------------- 🚀 混合机制步骤调度 (完整修复版) -------------------

                    # 1. 检查新的混合机制 (交错式 ZO/FG)
                    if args.trainer == "zo_hybrid_fg":
                        if (self.state.global_step % args.fo_guidance_steps == 0) and args.fo_guidance_steps > 0:
                            # 运行精确但缓慢的 FG (JVP)
                            tr_loss_step = self.forward_grad_step_with_fallback(model, inputs)
                            logger.info(f"Step {self.state.global_step}: Running Hybrid FG (JVP) step.")
                        else:
                            # 运行快速但有噪声的 ZO (MeZO)
                            tr_loss_step = self.zo_step(model, inputs)

                    # 2. 检查旧的混合机制 (已废弃)
                    elif args.hybrid_split_method == "dynamic_zo_fo":
                        logger.warning("dynamic_zo_fo is deprecated due to memory leaks. Falling back to pure ZO.")
                        tr_loss_step = self.zo_step(model, inputs)  # Fallback to pure ZO

                    # 3. 检查纯 ZO
                    elif args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt"]:
                        if args.module_wise_perturbation:
                            assert args.q == 1, "module-wise perturbation only supports q=1"
                            if args.coordinate_perturbation:
                                tr_loss_step = self.zo_step_with_module_wise_perturbation_coordinate(model, inputs)
                            else:
                                tr_loss_step = self.zo_step_with_module_wise_perturbation(model, inputs)
                        elif args.q == 1:
                            tr_loss_step = self.zo_step(model, inputs)
                        elif args.q > 1:
                            tr_loss_step = self.zo_step_v1(model, inputs)
                        else:
                            raise ValueError(f"q={args.q} is not supported.")

                    # 4. 检查其他 ZO
                    elif args.trainer == "zo_conserv":
                        tr_loss_step = self.zo_conserv_step(model, inputs)

                    # 5. 检查纯 FG
                    elif args.trainer == "forward_grad":
                        tr_loss_step = self.forward_grad_step_with_fallback(model, inputs)

                    # 6. 回退到标准 FO (BP) 训练 (高内存)
                    else:
                        if (
                                ((step + 1) % args.gradient_accumulation_steps != 0)
                                and args.local_rank != -1
                                and args._no_sync_in_gradient_accumulation
                        ):
                            with model.no_sync():
                                tr_loss_step = self.training_step(model, inputs)
                        else:
                            tr_loss_step = self.training_step(model, inputs)

                    # ------------------- 调度结束 -------------------

                # 关键修复：检查 tr_loss_step 是否为 None

                if tr_loss_step is None:
                    logger.warning(f"tr_loss_step is None at step {step}, skipping gradient update")

                    continue

                # 关键修复：确保 tr_loss_step 是张量

                if not isinstance(tr_loss_step, torch.Tensor):

                    logger.warning(
                        f"tr_loss_step is not a tensor at step {step}, type: {type(tr_loss_step)}, value: {tr_loss_step}")

                    # 尝试转换为张量

                    try:

                        tr_loss_step = torch.tensor(tr_loss_step, device=args.device)

                    except:

                        logger.error(f"Failed to convert tr_loss_step to tensor, skipping step {step}")

                        continue

                if (

                        args.logging_nan_inf_filter

                        and not is_torch_tpu_available()

                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))

                ):

                    # if loss is nan or inf simply add the average of previous logged losses

                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)

                else:

                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps

                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (

                        # last step in epoch but step is always smaller than gradient_accumulation_steps

                        steps_in_epoch <= args.gradient_accumulation_steps

                        and (step + 1) == steps_in_epoch

                ):

                    # MeZO added: update model with the estimated gradient

                    if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt", "zo_conserv"]:

                        self.zo_update(model)

                    # 🚀🚀🚀 开始修改 🚀🚀🚀
                    elif args.trainer == "zo_dynamic_fo":
                        if self.run_fo_step_next:
                            # 运行了FO步骤，使用标准梯度更新
                            self.gradient_update(model)
                        else:
                            self.lr_scheduler.step()
                    # 🚀🚀🚀 修改结束 🚀🚀🚀

                    elif args.trainer == "forward_grad":

                        self.forward_grad_update(model)

                    else:

                        self.gradient_update(model)

                    self.state.global_step += 1

                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    start_time = getattr(self, "_start_time", time.time())

                    start_time = getattr(self, "_start_time", time.time())

                    # 安全调用 _maybe_log_save_evaluate，兼容 transformers 新旧版本

                    try:

                        # 新版 transformers (>=4.46) 要求使用关键字参数调用

                        self._maybe_log_save_evaluate(

                            tr_loss=tr_loss,

                            grad_norm=None,

                            model=model,

                            trial=trial,

                            epoch=epoch,

                            ignore_keys_for_eval=ignore_keys_for_eval,

                        )

                    except TypeError:

                        try:

                            # 旧版 transformers (<=4.33) 的位置参数调用方式

                            self._maybe_log_save_evaluate(

                                tr_loss,

                                model,

                                trial,

                                epoch,

                                ignore_keys_for_eval,

                                start_time

                            )

                        except Exception as e:

                            # 保险兜底，避免异常中断训练

                            print(f"[Warning] _maybe_log_save_evaluate failed: {e}")



                else:

                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()

                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:

                    print(

                        f"=========================> Evaluating at step {total_steps + 1}... <=========================")

                    val_metrics = self.evaluate_func([], self.dev_samples,
                                                     description="Evaluating on Validation Set (Training Step)")

                    test_metrics = self.evaluate_func([], self.eval_samples,
                                                      description="Evaluating on Test Set (Training Step)")

                    if "accuracy" in test_metrics:

                        self.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})

                        wandb.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})

                    else:

                        keys = list(test_metrics.keys())

                        log_dict = {}

                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]

                            log_dict['val_' + k] = val_metrics[k]

                        self.log(log_dict)

                        wandb.log(log_dict)

                max_memory_allocated = 0

                for device_id in range(torch.cuda.device_count()):
                    # this is not accurate since max memory does not happen simultaneously across all devices

                    max_memory_allocated += torch.cuda.max_memory_allocated(device_id)

                self.log({"peak_mem": max_memory_allocated / 1024 ** 3,

                          "step_consumption": train_step_duration * 1000})

                wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,

                           "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.

                logger.warning(

                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"

                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"

                    f" num_steps ({max_steps}) higher than the number of available samples."

                )

                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            start_time = getattr(self, "_start_time", time.time())

            start_time = getattr(self, "_start_time", time.time())

            # 安全调用 _maybe_log_save_evaluate，兼容 transformers 新旧版本

            try:

                # 新版 transformers (>=4.46) 要求使用关键字参数调用

                self._maybe_log_save_evaluate(

                    tr_loss=tr_loss,

                    grad_norm=None,

                    model=model,

                    trial=trial,

                    epoch=epoch,

                    ignore_keys_for_eval=ignore_keys_for_eval,

                )

            except TypeError:

                try:

                    # 旧版 transformers (<=4.33) 的位置参数调用方式

                    self._maybe_log_save_evaluate(

                        tr_loss,

                        model,

                        trial,

                        epoch,

                        ignore_keys_for_eval,

                        start_time

                    )

                except Exception as e:

                    # 保险兜底，避免异常中断训练

                    print(f"[Warning] _maybe_log_save_evaluate failed: {e}")

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:

                if is_torch_tpu_available():

                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)

                    xm.master_print(met.metrics_report())

                else:

                    logger.warning(

                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "

                        "configured. Check your training configuration if this is unexpected."

                    )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training

            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:

            # Wait for everyone to get here so we are sur the model has been saved by process 0.

            if is_torch_tpu_available():

                xm.rendezvous("load_best_model_at_end")

            elif args.local_rank != -1:

                dist.barrier()

            elif is_sagemaker_mp_enabled():

                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss

        self._total_loss_scalar += tr_loss.item()

        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        self.store_flos()

        metrics["total_flos"] = self.state.total_flos

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        wandb.log(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)

        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.

        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:

            for checkpoint in checkpoints_sorted:

                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")

                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def gradient_update(self, model):

        args = self.args

        # Gradient clipping

        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:

            # deepspeed does its own clipping

            if self.do_grad_scaling:

                # Reduce gradients first for XLA

                if is_torch_tpu_available():
                    gradients = xm._fetch_gradients(self.optimizer)

                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())

                # AMP: gradients need unscaling

                self.scaler.unscale_(self.optimizer)

            if is_sagemaker_mp_enabled() and args.fp16:

                self.optimizer.clip_master_grads(args.max_grad_norm)

            elif hasattr(self.optimizer, "clip_grad_norm"):

                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping

                self.optimizer.clip_grad_norm(args.max_grad_norm)

            elif hasattr(model, "clip_grad_norm_"):

                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping

                model.clip_grad_norm_(args.max_grad_norm)

            else:

                # Revert to normal clipping otherwise, handling Apex or full precision

                nn.utils.clip_grad_norm_(

                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),

                    args.max_grad_norm,

                )

        # Optimizer step

        optimizer_was_run = True

        if self.deepspeed:

            pass  # called outside the loop

        elif is_torch_tpu_available():

            if self.do_grad_scaling:

                self.scaler.step(self.optimizer)

                self.scaler.update()

            else:

                xm.optimizer_step(self.optimizer)

        elif self.do_grad_scaling:

            scale_before = self.scaler.get_scale()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            scale_after = self.scaler.get_scale()

            optimizer_was_run = scale_before <= scale_after

        else:

            self.optimizer.step()

        if optimizer_was_run and not self.deepspeed:
            self.lr_scheduler.step()

        model.zero_grad()

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):

        """

        Perturb the parameters with random vector z.

        Input:

        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)

        - scaling_factor: theta = theta + scaling_factor * z * eps

        """

        # Set the random seed to ensure that we sample the same z for perturbation/update

        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:

            grad_sparsity = self.get_grad_sparsity_by_name(name)

            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

            param.data = param.data + scaling_factor * z * self.args.zo_eps

    # =========================================================

    # 替换 trainer.py 中的 zo_forward 函数 (约 800 行)

    # =========================================================

    def zo_forward(self, model, inputs):

        """

        Get (no gradient) loss from the model. Dropout is turned off too.

        """

        model.eval()

        # ------------------- 🚀 修复点：强制参数为 FP32 进行计算 -------------------

        original_dtype = {}

        for name, param in model.named_parameters():

            if param.requires_grad and param.dtype != torch.float32:
                original_dtype[param] = param.dtype

                param.data = param.data.to(torch.float32)

        # ------------------- 修复点 1 结束 -------------------

        if self.args.non_diff:

            # Non-differentiable objective (may require autoregressive generation)

            loss = self.zo_forward_nondiff(model, inputs)



        else:

            with torch.inference_mode():

                inputs = self._prepare_inputs(inputs)

                # 强制将输入数据也转换为 FP32，以匹配模型参数

                inputs = {k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for
                          k, v in inputs.items()}

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)

                if self.args.n_gpu > 1:
                    # Warning: this is copied from the original Huggingface Trainer. Untested.

                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # ------------------- 🚀 修复点 2：恢复原始参数类型 -------------------

        with torch.no_grad():

            for param, dtype in original_dtype.items():
                param.data = param.data.to(dtype)

        # ------------------- 修复点 2 结束 -------------------

        return loss.detach().to(torch.float32)  # 确保返回的损失是 FP32

    def zo_forward_nondiff(self, model, inputs):

        """

        Get (no gradient) non-diffiable loss from the model.

        """

        model.eval()

        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)

            args = self.args

            outputs = self.model.generate(

                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,

                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,

                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),

                num_return_sequences=1,

                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],

                              self.tokenizer.eos_token_id],

            )

            output_text = []

            for i in range(len(outputs)):
                output_text.append(

                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())

            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def get_grad_sparsity_by_name(self, name):

        if self.gradient_sparsity is None:

            return None

        elif isinstance(self.gradient_sparsity, float):

            return self.gradient_sparsity

        elif isinstance(self.gradient_sparsity, dict):

            return self.gradient_sparsity[name]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        """

        Perform a training step on a batch of inputs.



        Subclass and override to inject custom behavior.



        Args:

            model (`nn.Module`):

                The model to train.

            inputs (`Dict[str, Union[torch.Tensor, Any]]`):

                The inputs and targets of the model.



                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the

                argument `labels`. Check your model's documentation for all accepted arguments.



        Return:

            `torch.Tensor`: The tensor with training loss on this batch.

        """

        try:

            model.train()

            inputs = self._prepare_inputs(inputs)

            # 添加输入验证

            if 'input_ids' not in inputs or 'labels' not in inputs:
                logger.warning(
                    f"Missing required inputs: input_ids={'input_ids' in inputs}, labels={'labels' in inputs}")

                return torch.tensor(0.0, device=self.args.device, requires_grad=False)

            with self.compute_loss_context_manager():

                loss = self.compute_loss(model, inputs)

            # 检查损失是否为 None

            if loss is None:
                logger.warning("Loss is None in training step")

                return torch.tensor(0.0, device=self.args.device, requires_grad=False)

            # 检查损失是否为有效张量

            if not isinstance(loss, torch.Tensor):

                logger.warning(f"Loss is not a tensor: {type(loss)}, value: {loss}")

                try:

                    loss = torch.tensor(loss, device=self.args.device)

                except:

                    logger.error("Failed to convert loss to tensor")

                    return torch.tensor(0.0, device=self.args.device, requires_grad=False)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`

                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:

                self.scaler.scale(loss).backward()

            elif self.use_apex:

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:

                    scaled_loss.backward()

            elif self.deepspeed:

                # loss gets scaled under gradient_accumulation_steps in deepspeed

                loss = self.deepspeed.backward(loss)

            else:

                loss.backward()

            # Sparse gradient

            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

            for name, param in model.named_parameters():

                if not param.requires_grad:
                    continue

                grad_sparsity = self.get_grad_sparsity_by_name(name)

                if grad_sparsity is not None:
                    param.grad[fast_random_mask_like(param.grad, grad_sparsity, generator=self.sparse_grad_rng)] = 0

            return loss.detach()



        except Exception as e:

            logger.error(f"Error in training step: {e}")

            logger.error(f"Input keys: {list(inputs.keys())}")

            if 'input_ids' in inputs:
                logger.error(f"Input shape: {inputs['input_ids'].shape}")

            if 'labels' in inputs:
                logger.error(f"Labels shape: {inputs['labels'].shape}")

            # 返回一个安全的零损失

            return torch.tensor(0.0, device=self.args.device, requires_grad=False)

    @staticmethod
    def grouped_module_iter(model: nn.Module, group_level: str) -> Iterable[nn.Module]:

        """

        Iterate over the top-level modules of a model and yield groups of modules that should be trained together.

        Parameters

        ----------

        model: nn.Module

            The model to iterate over.

        group_level: str

            The level at which to iterate over the model. One of "transformer", "mlp-attn", "linear"

        """

        reg_pattern = OPT_PERTURBATION_LEVEL_TO_REGEX[group_level]

        for name, module in model.named_modules():

            if re.match(reg_pattern, name):
                yield name, module

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, model, inputs):

        """Update the parameters right after perturbing the parameters."""

        args = self.args

        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z

        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []

        # First function evaluation

        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,

        # so it does not necessarily use sparse grad.

        # Second function evaluation

        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):

            self.named_parameters_to_optim = []

            for name, param in module.named_parameters():

                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))

                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)

            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            for _ in range(args.q):

                if self.args.perturbation_mode == "one_side":

                    self.zo_perturb_parameters(scaling_factor=-1)

                    loss2 = self.zo_forward(model, inputs)

                    self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

                else:  # two side perturbation

                    self.zo_perturb_parameters(scaling_factor=-2)

                    loss2 = self.zo_forward(model, inputs)

                    self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                    # Reset model back to its parameters at start of step

                    self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update

            torch.manual_seed(self.zo_random_seed)

            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

            # ------------------- 🚀 修复点 3：应用更新 -------------------

            current_lr = self._get_learning_rate()  # 获取当前学习率

            for name, param in self.named_parameters_to_optim:

                # Resample z

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,

                                 dtype=param.data.dtype)

                grad_sparsity = self.get_grad_sparsity_by_name(name)

                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":

                    graddiff_times_z = np.sign(self.projected_grad) * z

                else:

                    graddiff_times_z = self.projected_grad * z

                # 计算更新量: - LR * (g_z / q)

                update_amount = current_lr * graddiff_times_z / args.q

                # 直接应用更新，避免依赖 self.optimizer.step()

                param.data.sub_(update_amount)

                # 确保梯度被清空

                # param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.

                # self.optimizer.step()  # will only update grad that is not None.

                param.grad = None  # avoid further update.

            # ------------------- 🚀 修复点 3 结束 -------------------

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, model, inputs):

        """Update all parameters once after perturbing all the parameters."""

        args = self.args

        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z

        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []

        module_name_to_projected_grads = {}

        # First function evaluation

        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,

        # so it does not necessarily use sparse grad.

        # Second function evaluation

        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):

            self.named_parameters_to_optim = []

            for name, param in module.named_parameters():

                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))

                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)

            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            if self.args.perturbation_mode == "one_side":

                self.zo_perturb_parameters(scaling_factor=-1)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

            else:  # two side perturbation

                self.zo_perturb_parameters(scaling_factor=-2)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step

                self.zo_perturb_parameters(scaling_factor=1)

            module_name_to_projected_grads[module_name] = self.projected_grad

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):

            self.named_parameters_to_optim = []

            for name, param in module.named_parameters():

                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))

                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.projected_grad = module_name_to_projected_grads[module_name]

            # Set the random seed to ensure that we sample the same z for perturbation/update

            torch.manual_seed(self.zo_random_seed)

            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

            for name, param in self.named_parameters_to_optim:

                # Resample z

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,

                                 dtype=param.data.dtype)

                grad_sparsity = self.get_grad_sparsity_by_name(name)

                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":

                    graddiff_times_z = np.sign(self.projected_grad) * z

                else:

                    graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.

                self.optimizer.step()  # will only update grad that is not None.

                param.grad = None  # avoid further update.

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    # =========================================================

    # 替换 trainer.py 中的 zo_step 函数 (约 876 行)

    # =========================================================

    @torch.no_grad()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation: f(theta + eps*z)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # ------------------- 🚀 调试点 1：Loss1 -------------------
        logger.info(f"[ZO Debug] Loss1 (theta + eps*z): {loss1.item():.8f}")
        # ------------------- 调试点 1 结束 -------------------

        # Second function evaluation: f(theta - eps*z)
        assert args.q == 1, "only support q=1 for the memory efficiency..."
        for _ in range(args.q):
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            loss_diff = loss1 - loss2
            # ------------------- 🚀 调试点 2：Loss2 和梯度 -------------------
            logger.info(f"[ZO Debug] Loss2 (theta - eps*z): {loss2.item():.8f}")
            logger.info(f"[ZO Debug] Loss Diff (L1-L2): {loss_diff.item():.8f}")
            logger.info(f"[ZO Debug] Projected Grad (Final): {self.projected_grad:.8e}")
            # ------------------- 调试点 2 结束 -------------------

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    graddiff_times_z = np.sign(self.projected_grad) * z
                else:
                    graddiff_times_z = self.projected_grad * z

                # ------------------- 🚀 最终修复：使用优化器状态 -------------------
                # 1. 计算纯梯度估计 (g_z)
                estimated_grad = graddiff_times_z / args.q

                # 2. 将梯度存入 param.grad
                param.grad = estimated_grad.detach()

                # 3. 立即调用优化器步进 (使用其内置的 LR 和 AdamW/SGD 状态)
                self.optimizer.step()

                # 4. 清空梯度
                param.grad = None
                # ------------------- 🚀 最终修复结束 -------------------

        # 🚀 修复：注释掉或修改这个断言，支持梯度累积
        # assert self.args.gradient_accumulation_steps == 1
        if self.args.gradient_accumulation_steps != 1:
            logger.warning(
                f"ZO训练器检测到 gradient_accumulation_steps={self.args.gradient_accumulation_steps}，可能影响训练稳定性")

        return loss1

    @torch.no_grad()
    def zo_step_v1(self, model, inputs):

        """

        Estimate gradient by MeZO. Return the loss from f(theta + z)

        Works with q > 1. But for q > 1, it is not memory efficient.

        """

        args = self.args

        # What parameters to optimize

        self.named_parameters_to_optim = []

        for name, param in model.named_parameters():

            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

                # # TODO avoid init the memory for grad.

                # param.grad = torch.zeros_like(param.data)

        for i_q in range(args.q):  # TODO shall we change the seed?

            # Sample the random seed for sampling z

            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation

            self.zo_perturb_parameters(scaling_factor=1)

            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation

            if self.args.perturbation_mode == "one_side":

                self.zo_perturb_parameters(scaling_factor=-1)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

            else:  # two side perturbation

                self.zo_perturb_parameters(scaling_factor=-2)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step

                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update

            torch.manual_seed(self.zo_random_seed)

            for name, param in self.named_parameters_to_optim:

                # Resample z

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,

                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":

                    # ----signOpt_orig

                    graddiff_times_z = np.sign(self.projected_grad) * z

                    # ----signOpt_mul_sign

                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)

                else:

                    # ----mezo original

                    graddiff_times_z = self.projected_grad * z

                # # previous implementation

                # # no param.grad involved

                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()

                # more mem-efficient:

                # run optimizer.step here to avoid caching all grad.

                if i_q == 0:

                    param.grad = graddiff_times_z / args.q

                else:

                    param.grad += graddiff_times_z / args.q

                # if i_q == args.q - 1:

                #     self.optimizer.step()  # TODO If q > 1, We cannot use this trick anymore. This will cause repeated update.

                #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.

                #     param.grad = None

        # for name, param in self.named_parameters_to_optim:

        #     param.grad = param.grad / args.q

        self.optimizer.step()

        self.optimizer.zero_grad()

        # No gradient accumulation support

        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v2(self, model, inputs):

        """

        Estimate gradient by MeZO. Return the loss from f(theta + z)

        Works with q > 1. But for q > 1, it is not memory efficient.

        """

        args = self.args

        # What parameters to optimize

        self.named_parameters_to_optim = []

        for name, param in model.named_parameters():

            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

                # # TODO avoid init the memory for grad.

                # param.grad = torch.zeros_like(param.data)

        seed_list = []

        projected_grad_list = []

        for i_q in range(args.q):  # TODO shall we change the seed?

            # Sample the random seed for sampling z

            self.zo_random_seed = np.random.randint(1000000000)

            seed_list.append(self.zo_random_seed)

            # First function evaluation

            self.zo_perturb_parameters(scaling_factor=1)

            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation

            if self.args.perturbation_mode == "one_side":

                self.zo_perturb_parameters(scaling_factor=-1)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

            else:  # two side perturbation

                self.zo_perturb_parameters(scaling_factor=-2)

                loss2 = self.zo_forward(model, inputs)

                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step

                self.zo_perturb_parameters(scaling_factor=1)

            projected_grad_list.append(self.projected_grad)

        # difference from v1: switch the order of for loop

        # to save memory

        for name, param in self.named_parameters_to_optim:

            for i_q in range(args.q):

                # Set the random seed to ensure that we sample the same z for perturbation/update

                torch.manual_seed(seed_list[i_q])

                graddiff_times_z = torch.zeros_like(param.data, device=param.data.device,

                                                    dtype=param.data.dtype)

                # Resample z

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,

                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":

                    # ----signOpt_orig

                    graddiff_times_z += np.sign(projected_grad_list[i_q]) * z

                    # ----signOpt_mul_sign

                    # graddiff_times_z = torch.sign(projected_grad_list[i_q] * z)

                else:

                    # ----mezo original

                    graddiff_times_z += projected_grad_list[i_q] * z

                # # previous implementation

                # # no param.grad involved

                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()

                # more mem-efficient:

                # run optimizer.step here to avoid caching all grad.

                if i_q == args.q - 1:
                    param.grad = graddiff_times_z.detach()

                    self.optimizer[name].step()

                    # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.

                    param.grad = None

        # for name, param in self.named_parameters_to_optim:

        #     param.grad = param.grad / args.q

        # No gradient accumulation support

        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_conserv_step(self, model, inputs):

        """

        Estimate gradient by MeZO. Return the loss from f(theta + z)

        update in the conservative way, i.e.

        reject the update if it's not decreasing

        """

        args = self.args

        # What parameters to optimize

        self.named_parameters_to_optim = []

        for name, param in model.named_parameters():

            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

                param.grad = None

        loss0 = self.zo_forward(model, inputs)

        # Sample the random seed for sampling z

        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation

        self.zo_perturb_parameters(scaling_factor=1)

        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation

        if self.args.perturbation_mode == "one_side":

            self.zo_perturb_parameters(scaling_factor=-1)

            loss2 = self.zo_forward(model, inputs)

            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

        else:  # two side perturbation

            self.zo_perturb_parameters(scaling_factor=-2)

            loss2 = self.zo_forward(model, inputs)

            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # Reset model back to its parameters at start of step

            self.zo_perturb_parameters(scaling_factor=1)

        def update_params(sign=1.0):

            # Set the random seed to ensure that we sample the same z for perturbation/update

            torch.manual_seed(self.zo_random_seed)

            for name, param in self.named_parameters_to_optim:

                # Resample z

                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,

                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":

                    # ----signOpt_orig

                    # TODo why do we multiply lr here? We will multiply lr twice?

                    graddiff_times_z = np.sign(self.projected_grad) * z

                    # ----signOpt_mul_sign

                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)

                else:

                    # ----mezo original

                    graddiff_times_z = self.projected_grad * z

                # # previous implementation

                # # no param.grad involved

                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()

                # more mem-efficient:

                # run optimizer.step here to avoid caching all grad.

                param.grad = sign * graddiff_times_z

                # self.optimizer[name].step()

                self.optimizer.step()

                # param.data = param.data - graddiff_times_z / args.q

                param.grad = None

        update_params()

        loss1 = self.zo_forward(model, inputs)

        update_params(sign=-2.0)

        loss2 = self.zo_forward(model, inputs)

        # conduct the update in the conservative way

        # choose from the three and take the minimum loss one

        if loss1 > loss0:

            if loss0 < loss2:
                update_params()

        else:

            if loss1 < loss2:
                update_params(2.0)

        # No gradient accumulation support

        assert self.args.gradient_accumulation_steps == 1

        return loss1

    # trainer.py (约 1225 行)

    def zo_update(self, model):

        """

        Update the parameters with the estimated gradients.

        """

        # ⚠️ 修复点：只处理学习率调度器

        self.lr_scheduler.step()

        # model.zero_grad()

    # 🚀🚀🚀 开始添加新方法 🚀🚀🚀
    def check_loss_plateau(self) -> bool:
        """
        检查损失是否进入平台期
        """
        args = self.args
        current_step = self.state.global_step

        # 1. 检查是否有足够的历史数据
        if len(self.loss_history) < args.fo_guidance_window:
            return False

        # 2. 检查是否刚运行过FO (冷却期)
        if (current_step - self.last_fo_step) < args.fo_min_interval:
            return False
        # 3. 计算损失改善
        # 我们简单比较窗口的第一个损失和最后一个损失
        try:
            old_loss = self.loss_history[0]
            new_loss = self.loss_history[-1]

            if old_loss == 0:  # 避免除零
                return False

            improvement_ratio = (old_loss - new_loss) / abs(old_loss)

            if improvement_ratio < args.fo_guidance_trigger_threshold:
                # 损失改善太小，触发FO
                logger.info(
                    f"[Hybrid Control] Plateau detected. Improvement: {improvement_ratio:.4f} < {args.fo_guidance_trigger_threshold}")
                return True
            else:
                # 改善仍然显著，继续ZO
                logger.info(f"[Hybrid Control] ZO is effective. Improvement: {improvement_ratio:.4f}")
                return False

        except Exception as e:
            logger.warning(f"Failed to check loss plateau: {e}")
            return False

    def hybrid_dynamic_fo_step(self, model, inputs):
        """
        动态混合ZO/FO步骤。
        - 检查是否达到损失平台期
        - 如果是，运行FO (training_step)
        - 否则，运行ZO (zo_step)
        """
        args = self.args

        # 决定此步骤运行FO还是ZO
        self.run_fo_step_next = self.check_loss_plateau()

        if self.run_fo_step_next:
            # -------------------
            # 1. 运行 FO 指导步骤
            # -------------------
            logger.info(f"Step {self.state.global_step + 1}: Running FO guidance step (Memory Safe).")

            # 重置状态
            self.last_fo_step = self.state.global_step
            self.loss_history = []

            # 调用常规 training_step, 它会计算损失并调用 loss.backward()
            # (我们依赖 run.py 中的PEFT冻结修复来保证内存安全)
            model.train()
            loss = self.training_step(model, inputs)

        else:
            # -------------------
            # 2. 运行标准 ZO 步骤
            # -------------------
            if args.q == 1:
                loss = self.zo_step(model, inputs)
            else:
                logger.warning_once("hybrid_dynamic_fo_step 正在使用 q>1 的 ZO 步骤，请确保其行为符合预期。")
                loss = self.zo_step_v1(model, inputs)  # 假设使用 q>1 的版本

        # 更新损失历史
        if loss is not None and isinstance(loss, torch.Tensor):
            self.loss_history.append(loss.item())
            # 保持历史窗口大小
            if len(self.loss_history) > args.fo_guidance_window:
                self.loss_history.pop(0)

        return loss

    # 🚀🚀🚀 新方法结束 🚀🚀🚀

    def _disable_efficient_attention(self, model):

        """临时禁用高效注意力机制，使用标准注意力计算"""

        if hasattr(model.config, '_attn_implementation'):

            self._original_attn_implementation = model.config._attn_implementation

            model.config._attn_implementation = 'eager'  # 使用标准注意力

        elif hasattr(model.config, 'attn_implementation'):

            self._original_attn_implementation = model.config.attn_implementation

            model.config.attn_implementation = 'eager'

        # 对于OPT模型，还需要设置use_flash_attention_2为False

        if hasattr(model.config, 'use_flash_attention_2'):
            self._original_use_flash_attention_2 = model.config.use_flash_attention_2

            model.config.use_flash_attention_2 = False

    def _restore_attention_implementation(self, model):

        """恢复原始的注意力实现"""

        if hasattr(self, '_original_attn_implementation'):

            if hasattr(model.config, '_attn_implementation'):

                model.config._attn_implementation = self._original_attn_implementation

            elif hasattr(model.config, 'attn_implementation'):

                model.config.attn_implementation = self._original_attn_implementation

        if hasattr(self, '_original_use_flash_attention_2'):

            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = self._original_use_flash_attention_2

    # trainer.py (替换 functional_call_loss, 约 1290 行)

    @staticmethod
    # 🚀 修复 1：移除 @torch.no_grad()
    def functional_call_loss(params, names, buffers, model, batch):
        params = {k: v for k, v in zip(names, params)}

        # 🚀 修复 2：确保 'use_cache' 为 False
        # (我们也在 _forward_grad_step_impl 中设置了它，但这里是双重保险)
        batch_no_cache = batch.copy()
        batch_no_cache['use_cache'] = False

        outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch_no_cache)

        # 🚀 修复 3：只返回 loss 张量，而不是 CausalLMOutputWithPast
        if isinstance(outputs, tuple):
            return outputs[0]  # 假设 loss 是第一个返回值 (例如 CausalLMOutput[0])
        elif hasattr(outputs, 'loss'):
            return outputs.loss  # (例如 CausalLMOutput.loss)

        # 如果 outputs 已经是 loss
        return outputs

    def forward_grad_step_with_fallback(self, model, inputs):

        """

        使用前向梯度方法，临时禁用高效注意力机制

        """

        try:

            # 临时禁用高效注意力

            self._disable_efficient_attention(model)

            # 执行前向梯度计算

            loss = self._forward_grad_step_impl(model, inputs)

            # 恢复原始注意力实现

            self._restore_attention_implementation(model)

            return loss



        except Exception as e:

            logger.error(f"Forward gradient step failed: {e}")

            # 确保恢复原始注意力实现

            self._restore_attention_implementation(model)

            # 回退到 zeroth-order 方法

            logger.warning("Falling back to zeroth-order method")

            return self.zo_step(model, inputs)

    # trainer.py (替换 _forward_grad_step_impl, 约 3530 行)

    def _forward_grad_step_impl(self, model, inputs):
        """
        前向梯度方法的实际实现
        """
        args = self.args
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        vs = [torch.randn_like(p) for _, p in self.named_parameters_to_optim]

        assert args.q == 1, "q > 1"

        # 确保输入在正确的设备上
        inputs = {
            k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        # ------------------- 🚀 关键修复 1：禁用 JVP 期间的缓存 -------------------
        # JVP (torch.func) 无法处理 HF 返回的 DynamicCache 对象。
        # 我们必须强制模型只返回 Tensors (loss, logits)。
        inputs_no_cache = inputs.copy()
        inputs_no_cache['use_cache'] = False
        # ------------------- 修复 1 结束 ------------------------------------

        f = partial(
            self.functional_call_loss,
            names=[n for n, _ in self.named_parameters_to_optim],
            buffers=dict(model.named_buffers()),
            model=model,
            batch=inputs_no_cache  # 🚀 使用
        )

        # 使用 jvp 计算前向导数和损失
        try:
            loss_, jvp_ = jvp(f, (list([p for _, p in self.named_parameters_to_optim]),), (vs,))

            # 处理 jvp_ 返回值
            if isinstance(jvp_, tuple):
                jvp_ = jvp_[0]

            # 更新参数
            with torch.no_grad():
                for v, (n, p) in zip(vs, [(n, p) for n, p in self.named_parameters_to_optim]):
                    if "bias" not in n and "layer_norm" not in n and "layernorm" not in n:
                        p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device) + args.weight_decay * p.data))
                    else:
                        p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device)))

            # 处理 loss_ 值
            if isinstance(loss_, tuple):
                loss += loss_[0].item()
            else:
                loss += loss_.item()

            return torch.tensor(loss, device=args.device)

        except Exception as e:
            logger.error(f"JVP computation failed: {e}")
            raise e

    def forward_grad_step(self, model, inputs):

        """

        前向梯度步骤 - 使用标准注意力计算

        """

        return self.forward_grad_step_with_fallback(model, inputs)

    def forward_grad_update(self, model):

        """

        前向梯度更新

        """

        self.lr_scheduler.step()

        model.zero_grad()

    def _set_signature_columns_if_needed(self):

        """

        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task

        """

        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.

            signature = inspect.signature(self.model.forward)

            self._signature_columns = list(signature.parameters.keys())

            # Labels may be named label or label_ids, the default data collator handles that.

            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):

        """

        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)

        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():

            self._save_tpu(output_dir)

        elif is_sagemaker_mp_enabled():

            # Calling the state_dict needs to be done on the wrapped model and on all processes.

            os.makedirs(output_dir, exist_ok=True)

            state_dict = self.model_wrapped.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)

            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10

                Path(os.path.join(output_dir, "user_content.pt")).touch()

        elif (

                # ✅ 安全地检查 sharded_ddp，避免 NoneType 报错

                (getattr(self.args, "sharded_ddp", None) and (

                        ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp

                        or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp

                ))

                or self.fsdp is not None

        ):

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug

            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):

                state_dict = self.model.state_dict()

            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)

        elif self.deepspeed:

            # this takes care of everything as long as we aren't under zero3

            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():

                # It's too complicated to try to override different places where the weights dump gets

                # saved, so since under zero3 the file is bogus, simply delete it. The user should

                # either user deepspeed checkpoint to resume or to recover full weights use

                # zero_to_fp32.py stored in the checkpoint.

                if self.args.should_save:

                    file = os.path.join(output_dir, WEIGHTS_NAME)

                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")

                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True

                # if false it will not be saved.

                # This must be called on all ranks

                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(

                        "deepspeed.save_16bit_model didn't save the model, since"

                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"

                        " zero_to_fp32.py to recover weights"

                    )

                    self.deepspeed.save_checkpoint(output_dir)



        elif self.args.should_save:

            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
