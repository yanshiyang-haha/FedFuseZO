import argparse
import os
import random
import time
import psutil
import GPUtil
from datetime import datetime
from resource_aware_hybrid import AdaptiveHybridTrainer
from federated_resource_aware_hybrid_trainer import FederatedHybridTrainer
import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from dataclasses import dataclass, field
from metrics import calculate_metric
# from modeling_mistral import (
#     MistralForCausalLM,
#     MistralConfig
# )
from tasks import get_task
from trainer import OurTrainer
from utils import *

os.environ["TRANSFORMERS_CACHE"] = "./cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# AutoConfig.register("mistral", MistralConfig)
# AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)


@dataclass
class OurArguments(TrainingArguments):
    # 🚀 资源感知混合训练参数
    resource_aware_training: bool = False  # 是否启用资源感知训练
    memory_threshold: float = 0.8  # 内存使用阈值 (0-1)
    adaptive_fo_ratio: float = 0.3  # 自适应FO比例
    min_compute_budget: float = 0.3  # 最小计算预算阈值
    min_bandwidth: float = 10.0  # 最小带宽阈值 (Mbps)
    # ========== Federated Learning Extra Arguments ==========
    num_clients: int = field(default=5, metadata={"help": "Number of federated clients"})
    participation_rate: float = field(default=0.6, metadata={"help": "Client participation rate per round"})
    aggregation_method: str = field(default="fedavg",
                                    metadata={"help": "Aggregation method (fedavg, median, trimmed_mean)"})
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 1000  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = False  # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = 0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo_sgd: zeroth-order SGD (MeZO) training
    ## - zo_conserv: zeroth-order SGD conservative training
    ## - zo_adam: zeroth-order Adam training
    ## - zo_sign_opt: zeroth-order sign sgd training
    ## - forward_grad: forward gradient
    optimizer: str = "adamw"
    ## options
    ## - sgd
    ## - adam
    ## - adamw # this is huggingface default
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification
    momentum: float = 0.0  # only work for SGD optimizer

    # 混合 ZO/FO 参数 🚀
    fo_guidance_trigger_threshold: float = 0.005  # 触发FO的损失改善阈值 (例如: 5步内改善小于0.5%)
    fo_guidance_window: int = 5  # 监测损失的窗口大小
    fo_min_interval: int = 10  # 两次FO步骤之间的最小间隔
    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "two_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = False  # whether to sample random tokens from Embedding layer

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    clean_model_at_end: bool = True  # remove everthing at the end.

    # sparse gradient pruning
    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"
    """
    Options
    ## - global: global sparsity will assign different sparsity to each layer, based on the pretrained weight magnitude
    ## - layer: each layer has the same sparsity
    """

    # module-wise perturbation
    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True  # If True, will update weight right after the gradient is computed
    """
    Options
    ## - transformer-block: perturb one transformer block at a time
    ## - mlp-attn: perturb one mlp/attention layer at a time
    ## - linear: perturb one linear layer at a time
    """
    # 🚀 动态混合 ZO/FO 参数
    trainer: str = "none"
    hybrid_split_method: str = "none"  # options: none, dynamic_zo_fo, fixed_ratio
    zo_fo_split_ratio: float = 0.5  # ZO 参数占总参数的比例
    fo_guidance_steps: int = 10  # 每 N 步运行一次 FO 指南 (BP)
    fo_guidance_trigger_threshold: float = 0.005  # 触发FO的损失改善阈值
    fo_guidance_window: int = 5  # 监测损失的窗口大小
    fo_min_interval: int = 10  # 两次FO步骤之间的最小间隔

    # 🎯 客户端资源感知参数
    resource_aware_training: bool = False  # 是否启用资源感知训练
    min_compute_budget: float = 0.3  # 最小计算预算阈值
    min_bandwidth: float = 10.0  # 最小带宽阈值 (Mbps)

    # 📊 自适应决策参数
    adaptive_decision_interval: int = 50  # 自适应决策间隔步数
    performance_decay_threshold: float = 0.1  # 性能衰减阈值


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PerformanceMetrics:
    """性能指标收集器"""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def record_metric(self, name, value):
        self.metrics[name] = value

    def get_system_metrics(self):
        """获取系统资源使用情况"""
        try:
            # GPU信息
            gpus = GPUtil.getGPUs()
            gpu_memory = sum([gpu.memoryUsed for gpu in gpus]) if gpus else 0

            # CPU和内存信息
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            ram_usage_gb = memory.used / (1024 ** 3)

            return {
                "gpu_memory_mb": gpu_memory,
                "cpu_usage_percent": cpu_percent,
                "ram_usage_gb": ram_usage_gb,
                "total_training_time": time.time() - self.start_time
            }
        except Exception as e:
            logger.warning(f"获取系统指标失败: {e}")
            return {}

    def calculate_perplexity(self, model, tokenizer, texts):
        """计算困惑度"""
        try:
            total_loss = 0
            total_tokens = 0

            model.eval()
            with torch.no_grad():
                for text in texts[:10]:  # 只计算前10个样本
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)

            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            return perplexity
        except Exception as e:
            logger.warning(f"计算困惑度失败: {e}")
            return 0.0

    def generate_summary_report(self, task_name, final_metrics, training_args):
        """生成详细的训练结果摘要"""
        system_metrics = self.get_system_metrics()

        print('\n' + '=' * 80)
        print('📊 ZO-LLM 训练结果详细摘要')
        print('=' * 80)

        print(f'\n🎯 任务信息:')
        print(f'  任务名称: {task_name}')
        print(f'  模型: {training_args.model_name}')
        print(f'  训练器: {training_args.trainer}')
        print(f'  训练样本数: {training_args.num_train}')

        print(f'\n⏱️ 训练性能:')
        print(f'  总训练时间: {system_metrics.get("total_training_time", 0):.1f}s')
        print(f'  学习率: {training_args.learning_rate:.2e}')
        print(f'  训练步数: {training_args.max_steps}')
        print(f'  Batch Size: {training_args.per_device_train_batch_size}')

        print(f'\n📈 准确率指标:')
        for key, value in final_metrics.items():
            if 'accuracy' in key.lower() or 'f1' in key.lower():
                print(f'  {key}: {float(value):.4f}')

        print(f'\n💾 系统资源:')
        print(f'  GPU显存使用: {system_metrics.get("gpu_memory_mb", 0):.0f} MB')
        print(f'  CPU使用率: {system_metrics.get("cpu_usage_percent", 0):.1f}%')
        print(f'  内存占用: {system_metrics.get("ram_usage_gb", 0):.2f} GB')

        print(f'\n⚙️ 训练配置:')
        print(f'  优化器: {training_args.optimizer}')
        print(f'  学习率调度器: {training_args.lr_scheduler_type}')
        print(f'  梯度累积步数: {training_args.gradient_accumulation_steps}')
        print(f'  最大序列长度: {training_args.max_length}')

        if hasattr(training_args, 'zo_eps'):
            print(f'\n🔬 MeZO配置:')
            print(f'  ZO Epsilon: {training_args.zo_eps}')
            print(f'  扰动模式: {training_args.perturbation_mode}')
            print(f'  采样数(Q): {training_args.q}')

        print(f'\n📊 其他指标:')
        for key, value in final_metrics.items():
            if 'accuracy' not in key.lower() and 'f1' not in key.lower() and key != 'improvement':
                print(f'  {key}: {value}')

        if 'improvement' in final_metrics:
            print(f'\n📈 训练改进:')
            for key, value in final_metrics['improvement'].items():
                trend = "↑" if value > 0 else "↓" if value < 0 else "→"
                print(f'  {key}: {value:+.4f} {trend}')

        print('\n' + '=' * 80)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.performance_metrics = PerformanceMetrics()
        self.model, self.tokenizer = self.load_model()
        self.training_start_time = None

        # 🚀 初始化混合训练器
        self.setup_hybrid_training()

    def setup_hybrid_training(self):
        """设置混合训练"""
        if self.args.hybrid_split_method != "none":
            from hybrid_trainer import AdaptiveHybridTrainer
            self.hybrid_trainer = AdaptiveHybridTrainer(
                self.args, self.model, self.tokenizer, self.performance_metrics
            )
            logger.info(f"🚀 启用混合训练模式: {self.args.hybrid_split_method}")
        else:
            self.hybrid_trainer = None

    def load_model(self):
        """
        Load HuggingFace models
        """
        load_start_time = time.time()
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            print(f"可用GPU内存: {free_in_GB}GB")
            config = AutoConfig.from_pretrained(self.args.model_name)

            # 检查模型类型
            model_type = config.model_type.lower()

            # 对于编码器模型（如RoBERTa、BERT），使用AutoModelForSequenceClassification
            if model_type in ["roberta", "bert", "distilbert", "albert"]:
                logger.info(f"检测到编码器模型 {model_type}，使用序列分类")
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch.float16 if self.args.load_float16 else torch.float32,
                    num_labels=2  # SST2是二分类任务
                )
            else:
                # 对于解码器模型（如GPT、OPT、LLaMA），使用原来的逻辑
                if self.args.untie_emb:
                    logger.warn("Untie embeddings and LM head")
                    config.tie_word_embeddings = False

                if self.args.head_tuning:
                    torch_dtype = torch.float32
                    if self.args.load_float16:
                        torch_dtype = torch.float16
                    elif self.args.load_bfloat16:
                        torch_dtype = torch.bfloat16
                    # Head tuning
                    if "opt" in self.args.model_name.lower():
                        from modeling_opt import OPTForCausalLM
                        model = OPTForCausalLM.from_pretrained(
                            self.args.model_name,
                            config=config,
                            device_map='auto',
                            torch_dtype=torch_dtype,
                            max_memory={i: f'{free_in_GB - 5}GB' for i in
                                        range(torch.cuda.device_count())},
                        )
                    elif "llama" in self.args.model_name.lower():
                        from modeling_llama import LlamaForCausalLMWithHeadTuning
                        model = LlamaForCausalLMWithHeadTuning.from_pretrained(
                            self.args.model_name,
                            config=config,
                            device_map='auto',
                            torch_dtype=torch_dtype,
                            max_memory={i: f'{free_in_GB - 5}GB' for i in
                                        range(torch.cuda.device_count())},
                        )
                    elif "mistral" in self.args.model_name.lower():
                        from modeling_mistral import MistralForCausalLMWithHeadTuning
                        model = MistralForCausalLMWithHeadTuning.from_pretrained(
                            self.args.model_name,
                            config=config,
                            device_map='auto',
                            torch_dtype=torch_dtype,
                            max_memory={i: f'{free_in_GB - 5}GB' for i in
                                        range(torch.cuda.device_count())},
                        )
                    else:
                        raise NotImplementedError(f"Head tuning is not supported for {self.args.model_name}")
                elif self.args.no_auto_device:
                    model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config)
                else:
                    torch_dtype = torch.float32
                    if self.args.load_float16:
                        torch_dtype = torch.float16
                    elif self.args.load_bfloat16:
                        torch_dtype = torch.bfloat16
                    model = AutoModelForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB - 5}GB' for i in range(torch.cuda.device_count())},
                        load_in_8bit=self.args.load_int8
                    )

            model.eval()

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # 对于编码器模型，需要设置padding token
        model_type = config.model_type.lower() if hasattr(config, 'model_type') else "unknown"
        if model_type in ["roberta", "bert", "distilbert", "albert"]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
            logger.info(f"设置编码器模型的padding token: {tokenizer.pad_token}")

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        if ("llama" in self.args.model_name) or ("mistral" in self.args.model_name.lower()):
            tokenizer.pad_token_id = 0

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix_tuning import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam,
                         float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            logger.info("🔧 初始化LoRA适配器...")
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

            # 统计LoRA参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"🔧 LoRA参数统计 - 总参数: {total_params:,}, 可训练参数: {trainable_params:,}, 比例: {trainable_params / total_params * 100:.4f}%")

            # 详细打印可训练参数
            if self.args.verbose:
                logger.info("🔧 可训练参数详情:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(f"  {name}: {param.numel():,}")

        if self.args.prompt_tuning:
            from prompt_tuning import PromptTuning
            print("Adding Prompt Tuning to model...")
            PromptTuning(
                model,
                num_virtual_tokens=self.args.num_virtual_tokens,
                init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
            )
            print("Total/Trainable number of parameters: {}/{}".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            ))

        if self.args.head_tuning:
            # ... (head_tuning 逻辑不变) ...
            if model.config.model_type in ["opt", "llama", "mistral"]:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        # ------------------- 🚨 关键修复：强制 PEFT 冻结 🚨 -------------------
        # 这一步旨在防止 LLM 主体参数（如 Transformer blocks）意外地保持 requires_grad=True，从而导致内存泄漏。

        is_peft_active = self.args.prompt_tuning or self.args.lora or self.args.prefix_tuning or self.args.head_tuning

        if is_peft_active:
            # 记录所有非 PEFT 模块（即 LLM 主体）的参数数量
            peft_trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

            total_frozen = 0

            for name, param in model.named_parameters():
                if name not in peft_trainable_names:
                    # 强制冻结 LLM 主体参数
                    param.requires_grad = False
                    total_frozen += param.numel()

            if total_frozen > 0:
                logger.info(f"🔒 [PEFT 内存检查] 最终强制冻结 LLM 主体参数: {total_frozen:,} 个。")

        # ------------------- 修复结束 ------------------------------------

        load_time = time.time() - load_start_time
        self.performance_metrics.record_metric("model_load_time", load_time)
        logger.info(f"模型加载时间: {load_time:.2f}s")

        return model, tokenizer

    def calculate_additional_metrics(self, eval_samples, description="Additional Metrics"):
        """计算额外的评估指标"""
        logger.info(f"=== {description} ===")

        metrics = {}

        # 计算推理速度
        inference_times = []
        for i, sample in enumerate(eval_samples[:10]):  # 只测试前10个样本
            start_time = time.time()
            _ = self.one_step_pred([], sample, verbose=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        if inference_times:
            avg_inference_time = np.mean(inference_times)
            metrics["avg_inference_time_per_sample"] = avg_inference_time
            metrics["samples_per_second"] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            logger.info(f"平均推理时间: {avg_inference_time:.4f}s/样本")
            logger.info(f"推理速度: {metrics['samples_per_second']:.2f} 样本/秒")

        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        metrics["total_parameters"] = total_params
        metrics["trainable_parameters"] = trainable_params
        metrics["trainable_ratio"] = trainable_params / total_params if total_params > 0 else 0

        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")
        logger.info(f"可训练参数比例: {metrics['trainable_ratio']:.4f}")

        return metrics

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        # 检查模型类型
        model_type = self.model.config.model_type.lower() if hasattr(self.model.config, 'model_type') else "unknown"

        if model_type in ["roberta", "bert", "distilbert", "albert"]:
            # 编码器模型 - 直接分类
            with torch.inference_mode():
                self.model.eval()
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

                # 对于分类任务，返回两个类别的log probabilities
                log_probs = F.log_softmax(logits, dim=-1)[0]  # [batch_size, num_classes]

                # 对于编码器模型，我们直接返回分类logits，不处理option_len
                return log_probs.cpu().detach()

        else:
            # 解码器模型 - 原来的逻辑
            if generation:
                args = self.args
                outputs = self.model.generate(input_ids, do_sample=args.sampling, temperature=args.temperature,
                                              num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                                              max_new_tokens=min(args.max_new_tokens,
                                                                 args.max_length - input_ids.size(1)),
                                              num_return_sequences=1,
                                              eos_token_id=[
                                                  self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                                                  self.tokenizer.eos_token_id], )
                output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
                return output_text
            else:
                with torch.inference_mode():
                    self.model.eval()
                    logits = self.model(input_ids=input_ids).logits
                labels = input_ids[0, 1:]
                logits = logits[0, :-1]
                log_probs = F.log_softmax(logits, dim=-1)

                selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
                selected_log_probs = selected_log_probs.cpu().detach()
                return selected_log_probs[-option_len:] if option_len else selected_log_probs

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose

        # 检查模型类型
        model_type = self.model.config.model_type.lower() if hasattr(self.model.config, 'model_type') else "unknown"

        if model_type in ["roberta", "bert", "distilbert", "albert"]:
            # 编码器模型 - 直接分类逻辑
            try:
                # 这里需要实现编码器模型的编码逻辑
                # 暂时回退到解码器逻辑
                pass
            except:
                pass

        # --- 🚀 核心修改开始 ---
        # 动态调整 max_length 以预留给 Prompt Tuning 的虚拟 tokens
        adjusted_max_length = self.args.max_length
        if self.args.prompt_tuning:
            # 在推理阶段，也要为 Prompt 预留空间
            adjusted_max_length = self.args.max_length - self.args.num_virtual_tokens
        # --- 核心修改结束 ---

        # 解码器模型 - 原来的逻辑
        encoded_candidates, option_lens = encode_prompt(self.task,
                                                        self.task.get_template(template_version=self.args.template_ver),
                                                        train_samples, eval_sample,
                                                        self.tokenizer,
                                                        # ⬇️ 原始代码: max_length=self.args.max_length
                                                        max_length=adjusted_max_length,  # 🌟 修改点：使用调整后的最大长度
                                                        generation=self.task.generation,
                                                        max_new_tokens=self.args.max_new_tokens)

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(
                template_version=self.args.template_ver), train_samples,
                                                                    eval_sample, self.tokenizer,
                                                                    # ⬇️ 原始代码: max_length=self.args.max_length
                                                                    max_length=adjusted_max_length,  # 🌟 修改点：使用调整后的最大长度
                                                                    sfc=self.args.sfc,
                                                                    icl_sfc=self.args.icl_sfc,
                                                                    generation=self.task.generation,
                                                                    max_new_tokens=self.args.max_new_tokens)
        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            # if verbose:
            #     logger.info("=== Prompt ===")
            #     logger.info(self.tokenizer.decode(encoded_candidates[0]))
            #     logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    # if candidate_id == 0:
                    #     logger.info("=== Candidate %d ===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate))
                    # else:
                    #     logger.info("=== Candidate %d (without context)===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id],
                                                          option_len=sfc_option_lens[
                                                              candidate_id])  # if verbose:  #     logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)  #     logger.info(  #         self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])  #     logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs,
                                "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False, description=None):
        """
        Evaluate function.
        Here, train_samples are used for demonstrations for ICL.
        If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        Otherwise, the same training set is used for all eval samples.
        """
        # 检查评估样本
        if not eval_samples:
            logger.warning("评估样本为空，返回默认指标")
            metric_name = getattr(self.task, "metric_name", "accuracy")
            return {metric_name: 0.0}

        # 检查训练样本（对于ICL）
        if one_train_set_per_eval_sample:
            if len(train_samples) != len(eval_samples):
                logger.warning(f"训练样本数({len(train_samples)})与评估样本数({len(eval_samples)})不匹配")
        else:
            if not train_samples:
                logger.info("使用零样本评估（无训练样本作为demonstration）")

        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc=description)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                   eval_sample, verbose=False))

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def validate_training_data(self, train_samples, dev_samples, eval_samples):
        """验证训练数据的有效性"""
        logger.info("=== 训练数据验证 ===")

        issues = []

        # 检查训练样本
        if not train_samples:
            issues.append("训练样本为空")
        else:
            logger.info(f"训练样本数量: {len(train_samples)}")

            # 检查样本结构
            sample = train_samples[0]
            if not hasattr(sample, 'correct_candidate'):
                issues.append("训练样本缺少 correct_candidate 属性")
            if not hasattr(sample, 'candidates'):
                issues.append("训练样本缺少 candidates 属性")

        # 检查评估样本
        if not eval_samples:
            issues.append("评估样本为空")
        else:
            logger.info(f"评估样本数量: {len(eval_samples)}")

        # 检查开发样本
        if dev_samples and len(dev_samples) == 0:
            issues.append("开发样本为空")
        elif dev_samples:
            logger.info(f"开发样本数量: {len(dev_samples)}")

        if issues:
            logger.error("训练数据存在问题:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ 训练数据验证通过")
            return True

    def train(self, train_samples, dev_samples, eval_samples):
        """
        Training function
        """
        self.training_start_time = time.time()

        # 数据验证
        if not self.validate_training_data(train_samples, dev_samples, eval_samples):
            logger.error("训练数据验证失败，跳过训练")
            return train_samples, dev_samples

        logger.info(f"Eval sample length is {len(eval_samples)}")
        logger.info(f"Train sample length is {len(train_samples)}")
        logger.info(f"Dev sample length is {len(dev_samples) if dev_samples is not None else 0}")

        # 检查训练样本是否为空
        if len(train_samples) == 0:
            logger.error("训练样本为空！无法进行训练")
            return train_samples, dev_samples

        # 添加调试信息
        logger.info("=== 训练前调试信息 ===")
        logger.info(f"训练器类型: {self.args.trainer}")
        logger.info(f"学习率: {self.args.learning_rate}")
        logger.info(f"最大步数: {self.args.max_steps}")
        logger.info(f"Batch Size: {self.args.per_device_train_batch_size}")
        logger.info(f"优化器: {self.args.optimizer}")
        logger.info(f"仅训练选项: {self.args.only_train_option}")
        logger.info(f"分类训练: {self.args.train_as_classification}")

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"训练前可训练参数: {trainable_params:,}/{total_params:,} ({trainable_params / total_params * 100:.4f}%)")

        # 调试：检查模型在训练前的输出
        logger.info("=== Debug: Pre-training Model Outputs ===")
        self.debug_model_outputs(train_samples[:2], eval_samples[:2])

        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []

            # 检查输入样本
            if len(samples) == 0:
                logger.warning("_convert函数接收到空样本列表！")
                return data

            logger.info(f"正在转换 {len(samples)} 个样本")
            # --- 🚀 核心修改开始 ---
            # 动态调整 max_length 以预留给 Prompt Tuning 的虚拟 tokens
            adjusted_max_length = self.args.max_length
            if self.args.prompt_tuning:
                # 检查并从 max_length 中减去虚拟 token 数量，防止溢出
                prompt_len = self.args.num_virtual_tokens
                if prompt_len >= self.args.max_length:
                    logger.error(
                        f"❌ 虚拟 token 数量 ({prompt_len}) 超过或等于 max_length ({self.args.max_length})，无法训练！")
                    return []
                adjusted_max_length = self.args.max_length - prompt_len
                logger.info(
                    f"✨ Prompt Tuning 激活: 原始 max_length={self.args.max_length}, 调整后的输入最大长度={adjusted_max_length}")

            # --- 核心修改结束 ---

            for i, sample in enumerate(samples):
                # 调试前几个样本
                if i < 2:
                    logger.info(f"样本 {i} 类型: {type(sample)}")
                    logger.info(f"样本 {i} 内容: {sample}")

                try:
                    encoded_candidates, option_lens = encode_prompt(self.task, self.task.get_template(
                        template_version=self.args.template_ver), [], sample,
                                                                    self.tokenizer,
                                                                    # ⬇️ 原始代码: max_length=self.args.max_length
                                                                    max_length=adjusted_max_length,  # 🌟 修改点：使用调整后的最大长度
                                                                    generation=self.task.generation,
                                                                    generation_with_gold=True,
                                                                    max_new_tokens=self.args.max_new_tokens)

                    # 检查编码结果
                    if i < 2:
                        logger.info(f"样本 {i} 编码结果: {len(encoded_candidates)} 个候选")
                        logger.info(f"样本 {i} 选项长度: {option_lens}")

                except Exception as e:
                    logger.error(f"样本 {i} 编码失败: {e}")
                    continue  # 跳过这个样本，继续处理下一个

                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][
                                                               :-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        # 关键修复：正确设置标签，只对选项部分计算loss
                        input_ids = encoded_candidates[correct_candidate_id]
                        labels = input_ids.copy()

                        # 将非选项部分的标签设置为-100（在计算loss时忽略）
                        if option_lens[correct_candidate_id] > 0:
                            prefix_len = len(labels) - option_lens[correct_candidate_id]
                            labels = [-100] * prefix_len + labels[prefix_len:]

                        data.append({
                            "input_ids": input_ids,
                            "labels": labels,
                            "option_len": option_lens[correct_candidate_id]
                        })

                        # 调试标签设置
                        if i < 2:
                            logger.info(f"输入IDs长度: {len(input_ids)}")
                            logger.info(f"标签长度: {len(labels)}")
                            logger.info(f"选项长度: {option_lens[correct_candidate_id]}")
                            logger.info(f"输入解码: {self.tokenizer.decode(input_ids)}")
                            logger.info(f"非忽略标签: {[l for l in labels if l != -100]}")
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
            dev_dataset = HFDataset(_convert(dev_samples))

        # 检查数据集样本
        logger.info(f"训练数据集大小: {len(train_dataset)}")
        logger.info(f"评估数据集大小: {len(eval_dataset)}")
        logger.info(f"开发数据集大小: {len(dev_dataset)}")

        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"样本训练数据键: {list(sample.keys())}")
            if 'input_ids' in sample:
                logger.info(f"输入IDs长度: {len(sample['input_ids'])}")
                logger.info(f"标签长度: {len(sample['labels'])}")
                logger.info(f"选项长度: {sample.get('option_len', 'N/A')}")
                logger.info(f"输入解码: {self.tokenizer.decode(sample['input_ids'])}")
                logger.info(f"非忽略标签: {[l for l in sample['labels'] if l != -100]}")
        else:
            logger.error("训练数据集为空！")

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.gradient_sparsity is not None:
            logger.info(
                f"[Sparse gradient] sparsity is {self.args.gradient_sparsity}, resampling per {self.args.sparse_gradient_resample_steps} steps"
            )

            if self.args.sparse_gradient_group == "global":
                logger.info(f"[Sparse gradient] global-ratio random pruning is enabled, "
                            f"sparsity of each layer is computed based on the pretrained weight magnitude.")
            elif self.args.sparse_gradient_group == "layer":
                logger.info(f"[Sparse gradient] layer-wise random pruning is enabled, "
                            f"sparsity of each layer is the same.")
            else:
                raise NotImplementedError(f"Unknown sparse gradient group: {self.args.sparse_gradient_group}")

        perturb_module_regex = None
        if self.args.module_wise_perturbation:
            if "opt" in self.args.model_name:
                assert self.args.perturbed_module_level in OPT_PERTURBATION_LEVEL_TO_REGEX.keys(), f"Unknown perturbed module group {self.args.perturbed_module_level}"
                perturb_module_regex = OPT_PERTURBATION_LEVEL_TO_REGEX[self.args.perturbed_module_level]
            else:
                raise NotImplementedError(f"Unimplemented model {self.args.model_name} for module-wise perturbation")

        trainer = OurTrainer(model=self.model,
                             args=self.args,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             tokenizer=self.tokenizer,
                             data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                             pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                                 self.tokenizer, pad_to_multiple_of=8),
                             eval_samples=eval_samples,
                             dev_samples=dev_samples,
                             evaluate_func=self.evaluate,
                             perturb_module_regex=perturb_module_regex,
                             )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # 最终检查
        logger.info("=== 最终训练前检查 ===")
        final_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"最终可训练参数: {final_trainable_params:,}")

        if len(train_dataset) == 0:
            logger.error("❌ 训练数据集为空，无法训练")
            return train_samples, dev_samples
        else:
            logger.info("✅ 可训练参数检查通过，开始训练...")

        # This calls the trainer._inner_training_loop()
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.info("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

        # 调试：检查模型在训练后的输出
        logger.info("=== Debug: Post-training Model Outputs ===")
        self.debug_model_outputs(train_samples[:2], eval_samples[:2])

        return train_samples, dev_samples

    def debug_model_outputs(self, train_samples, eval_samples, num_samples=3):
        """
        调试模型输出，检查训练前后模型的行为变化
        """
        logger.info("=== Debug: Model Outputs ===")
        for i in range(min(num_samples, len(eval_samples))):
            eval_sample = eval_samples[i]
            logger.info(f"Sample {i}: {eval_sample}")

            # 检查模型预测
            prediction = self.one_step_pred(train_samples, eval_sample, verbose=True)
            logger.info(f"Prediction: {prediction}")

    def delete_checkpoints(self):
        import shutil
        print(f"\nWARNING: Removing everything at end: {self.args.output_dir}")
        deleted_folders = [folder for folder in os.listdir(self.args.output_dir)
                           if os.path.isdir(os.path.join(self.args.output_dir, folder))
                           and folder.startswith("checkpoint-")]
        for f in deleted_folders:
            shutil.rmtree(os.path.join(self.args.output_dir, f))
        print(f"deleted folders: ", deleted_folders)


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()

    # 添加参数检查和调整
    logger.info("=== Training Configuration ===")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Trainer: {args.trainer}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Max Steps: {args.max_steps}")
    logger.info(f"Batch Size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Only Train Option: {args.only_train_option}")
    logger.info(f"Train as Classification: {args.train_as_classification}")

    # 对于 MeZO 训练器，学习率应该更小
    if args.trainer in ["zo_sgd", "zo_adam", "zo_conserv", "zo_sign_opt", "adam"]:
        if args.learning_rate > 1e-5:
            logger.warning(f"Learning rate {args.learning_rate} is too high for MeZO training!")
            logger.warning("For MeZO, recommended learning rate is 1e-5 to 1e-7")
            logger.warning("Consider using --learning_rate 1e-5")

    if args.max_steps < 100:
        logger.warning(f"Max steps {args.max_steps} might be too low for effective training")
        logger.warning("Consider using --max_steps 1000")

    if args.prefix_tuning:
        args.mode = "prefix"
    elif args.lora:
        args.mode = "lora"
    elif args.prompt_tuning:
        args.mode = "prompt"
    else:
        args.mode = "ft"
    args.tag = f"{args.trainer}-{args.task_name}-{args.template_ver}-{args.model_name.split('/')[-1]}-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}-LR{args.learning_rate}-{args.lr_scheduler_type}-ZOEPS{args.zo_eps}-Q{args.q}"
    args.tag = "momen" + args.tag if args.momentum > 0 else args.tag
    args.tag = f"sparse_grad-{args.gradient_sparsity}-{args.sparse_gradient_group}-{args.sparse_gradient_resample_steps}-" + args.tag if args.gradient_sparsity is not None else args.tag
    args.tag = f"module_perturb-{args.perturbed_module_level}-" + args.tag if args.module_wise_perturbation else args.tag
    args.run_name = args.tag
    args.output_dir = f"result/{args.tag}"
    args.result_file = f"result/{args.tag}/results.json"
    os.makedirs(args.output_dir, exist_ok=True)
    args.logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.logging_dir, exist_ok=True)

    wandb.init(project='zo-bench', name=args.tag, config=args)

    set_seed(args.seed)
    task = get_task(args.task_name)

    # This function samples both training and validation samples. The validation (dev) samples are also stored in "train_sets"
    # Later the train_samples and dev_samples are separated
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval,
                                        num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:

        # Training goes to this way

        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                # Here the training samples are separated
                if args.num_dev is not None:
                    # 修复：检查样本数量是否足够分割
                    total_samples = len(train_samples)
                    if total_samples <= args.num_dev:
                        logger.error(f"训练样本数({total_samples})不足以分割出开发样本({args.num_dev})")
                        # 调整开发样本数量，确保有训练样本
                        args.num_dev = min(args.num_dev, total_samples - 1)  # 至少保留1个训练样本
                        if args.num_dev <= 0:
                            logger.error("没有足够的样本进行训练，跳过此训练集")
                            continue
                        logger.info(f"调整开发样本数为: {args.num_dev}")

                    dev_samples = train_samples[-args.num_dev:]
                    train_samples = train_samples[:-args.num_dev]
                    logger.info(f"开发样本: {len(dev_samples)}")
                    logger.info(f"训练样本: {len(train_samples)}")
                else:
                    dev_samples = None
                    logger.info(f"训练样本: {len(train_samples)}")
                    logger.info("无开发样本")

                # 再次检查训练样本是否为空
                if len(train_samples) == 0:
                    logger.error("训练样本分割后为空！跳过训练，进行零样本评估")
                    metrics = framework.evaluate([], eval_samples, description="Zero-shot Evaluation")

                    # 计算额外指标
                    additional_metrics = framework.calculate_additional_metrics(eval_samples)
                    metrics.update(additional_metrics)

                    # 生成摘要报告
                    framework.performance_metrics.generate_summary_report(
                        args.task_name, metrics, args
                    )
                else:
                    # 正常训练流程
                    args.dev_samples = dev_samples
                    args.eval_samples = eval_samples

                    # 训练前评估
                    logger.info("=== Pre-training Evaluation ===")
                    pre_train_metrics = framework.evaluate(
                        train_samples,
                        eval_samples,
                        description="Pre-training Evaluation"
                    )
                    logger.info(f"Pre-training metrics: {pre_train_metrics}")

                    # 计算训练前额外指标
                    pre_train_additional = framework.calculate_additional_metrics(
                        eval_samples, "Pre-training Additional Metrics"
                    )

                    # Training
                    train_samples, dev_samples = framework.train(train_samples,
                                                                 dev_samples if dev_samples is not None else eval_samples,
                                                                 eval_samples)

                    if not args.no_eval:
                        # 训练后评估 - 使用相同的评估设置
                        logger.info("=== Post-training Evaluation ===")
                        post_train_metrics = framework.evaluate(
                            train_samples,
                            eval_samples,
                            description="Post-training Evaluation"
                        )

                        # 计算训练后额外指标
                        post_train_additional = framework.calculate_additional_metrics(
                            eval_samples, "Post-training Additional Metrics"
                        )

                        metrics = post_train_metrics.copy()
                        _keys = list(post_train_metrics.keys())
                        for m in _keys:
                            metrics["test_" + m] = post_train_metrics[m]

                        if dev_samples is not None:
                            dev_metrics = framework.evaluate(
                                train_samples,
                                dev_samples,
                                description="Validation Evaluation"
                            )
                            _keys = list(dev_metrics.keys())
                            for m in _keys:
                                metrics["val_" + m] = dev_metrics[m]

                        # 合并所有指标
                        metrics.update(post_train_additional)

                        # 计算训练前后的改进
                        improvement = {}
                        for key in pre_train_metrics:
                            if key in post_train_metrics:
                                improvement[key] = float(post_train_metrics[key]) - float(pre_train_metrics[key])

                        logger.info(f"Training improvement: {improvement}")
                        metrics["improvement"] = improvement

                        # 生成详细的训练结果摘要
                        framework.performance_metrics.generate_summary_report(
                            args.task_name, metrics, args
                        )
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

                # 计算额外指标
                additional_metrics = framework.calculate_additional_metrics(eval_samples)
                metrics.update(additional_metrics)

                # 生成摘要报告
                framework.performance_metrics.generate_summary_report(
                    args.task_name, metrics, args
                )

            logger.info(metrics)
            wandb.log(metrics)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                wandb.log(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" + result_file_tag(
                        args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples
        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)

        # 计算额外指标
        additional_metrics = framework.calculate_additional_metrics(eval_samples)
        metrics.update(additional_metrics)

        # 生成摘要报告
        framework.performance_metrics.generate_summary_report(
            args.task_name, metrics, args
        )

        logger.info(metrics)
        wandb.log(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(
                args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)


if __name__ == "__main__":
    main()
