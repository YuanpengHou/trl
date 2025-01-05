# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# 导入dataclasses模块中的dataclass和field函数，用于创建数据类
# dataclass简化了类的创建过程，使得类更加清晰、简洁
# field用于在dataclass中定义特殊属性
from dataclasses import dataclass, field
# 导入typing模块中的Any, Optional, Union类型，用于类型注解
# Any表示可以是任意类型
# Optional表示其参数类型可以是可选的，即可以是None
# Union表示其参数类型可以是几种类型中的一种
from typing import Any, Optional, Union

# 导入评估模块，用于模型性能的评估
import evaluate
# 导入numpy库，用于数值计算
import numpy as np
# 导入torch库，用于深度学习模型的构建和训练
import torch
# 导入神经网络模块库
import torch.nn as nn
# 导入数据集加载模块，用于加载预定义的数据集
from datasets import load_dataset
# 导入PEFT库，这些模块主要用于配置和获取基于 LoRA（Low-Rank Adaptation）的模型，以便在特定任务上进行微调。
# LoraConfig：用于定义 LoRA 的配置参数。
# TaskType：定义了不同任务类型，如文本分类、命名实体识别等。
# get_peft_model：根据给定的配置获取一个预训练模型，并应用 LoRA 进行微调。
from peft import LoraConfig, TaskType, get_peft_model
# 导入transformers库中的多个模块和类
from transformers import AutoModelForSequenceClassification  # 用于序列分类的自动模型
from transformers import AutoTokenizer  # 自动分词器
from transformers import HfArgumentParser  # Hugging Face参数解析器
from transformers import PreTrainedTokenizerBase  # 预训练分词器基类
from transformers import Trainer  # 训练器
from transformers import TrainerCallback  # 训练器回调接口
from transformers import TrainingArguments  # 训练参数
from transformers import set_seed  # 设置随机种子函数
# 导入设置工具模块，用于处理填充策略等
from transformers.utils import PaddingStrategy


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    # 多 GPU 设置
    # 建议值：默认值 -1，在单 GPU 或多 GPU 分布式训练时由框架自动设置。
    # 理由：用于分布式训练中的设备编号，通常不需要手动设置。
    # local_rank 字段用于指定多GPU训练时的本地排名。
    # 默认值为 -1，表示未指定或单GPU模式。
    # 通过 metadata 提供帮助信息，方便用户理解字段用途。
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    # 恢复训练
    # 建议值：False
    # 理由：除非你有特定的检查点需要恢复，否则保持默认值即可。
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    # DeepSpeed 配置
    # 建议值：None
    # 理由：如果你的模型可以放在单个 GPU 上，或者不使用 DeepSpeed，保持默认值。如果需要处理更大的模型或数据集，可以配置 DeepSpeed。
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    # 批量大小
    # Training batch size per device
    # 建议值：8（对于 RTX 3090），16（对于 A100）
    # 理由：较大的批量大小可以加速训练，但需要确保 GPU 内存足够。
    per_device_train_batch_size: Optional[int] = field(default=4)
    # Evaluation batch size per device
    # 建议值：2（对于 RTX 3090），4（对于 A100）
    # 理由：评估阶段的批量大小可以适当增大，以提高评估速度。
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # 梯度累积步数
    # Number of gradient accumulation steps
    # 建议值：2（对于 RTX 3090），1（对于 A100）
    # 理由：当单个 GPU 内存不足时，可以通过梯度累积来模拟更大的批量大小。
    gradient_accumulation_steps: Optional[int] = field(default=1)
    # 学习率
    # Learning rate for the optimizer
    # 建议值：5e-5
    # 理由：较高的学习率有助于更快收敛，但需要根据具体任务调整。
    learning_rate: Optional[float] = field(default=2e-5)
    # 权重衰减
    # Weight decay for the optimizer
    # 建议值：0.01
    # 理由：适当的权重衰减可以防止过拟合，特别是在大数据集上。
    weight_decay: Optional[float] = field(default=0.001)
    # 模型选择
    # 建议值："gpt2-medium" 或 "bert-base-uncased"
    # 理由：根据任务需求选择合适的预训练模型，gpt2-medium 是一个平衡性能和资源消耗的选择。
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    # Tokenizer选择
    # 建议值：None，对于 gpt2-medium 模型，默认的分词器是 GPT2Tokenizer。
    # 理由：通常使用与模型匹配的默认 tokenizer 即可。
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    # BF16 支持
    # 建议值：True
    # 理由：BF16 可以显著加快训练速度，同时对精度影响较小，适用于支持 BF16 的 GPU。
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    # 训练轮数epochs
    # 建议值：3
    # 理由：更多的训练轮数有助于模型更好地收敛，但需要监控过拟合情况。
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    # 数据子集大小
    # 建议值：100000
    # 理由：根据你的数据集大小和训练时间限制进行调整。指的是从完整的训练数据集中抽取的一部分数据。使用数据子集可以减少训练时间、测试模型性能或进行调试。
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    # 建议值：50000
    # 理由：评估数据集大小应适中，既能反映模型性能又不会浪费过多计算资源。
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    # 梯度检查点： 通过在前向传播中仅保存部分激活值即“检查点”（而不是所有激活值），在反向传播阶段直接使用保存的激活值，但要重新计算未保存的激活值。利弊如下：
    # 减少显存占用：通过丢弃部分激活值。
    # 增加计算成本：在反向传播中重新计算丢弃的激活值。
    # 建议值：False
    # 理由：除非内存非常紧张，否则不启用，因为会增加训练时间。
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    # 优化器
    # 建议值："adamw_hf"
    # 理由：AdamW 是常用的优化器，适合大多数 NLP 任务。
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    # 学习率调度器。学习率调度器Learning Rate Scheduler是训练过程中用于动态调整优化器学习率的工具。通过在训练的不同阶段调整学习率，可以提高模型的收敛速度和最终性能。常见的学习率调度策略包括线性衰减Linear Scheduler、余弦退火Cosine Annealing Scheduler先快速下降再缓慢上升，最后再下降、指数衰减Exponential Decay Scheduler、阶梯式调度器Step Scheduler等。
    # 建议值："linear"
    # 理由：线性调度器简单有效，适合大多数任务。
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    # 最大长度
    # 建议值：512
    # 理由：适合大多数 NLP 任务，可以根据具体任务调整。
    max_length: Optional[int] = field(default=512)
    # 初始评估。初始评估是在训练过程的非常早期阶段（通常是第一个训练步骤之后）进行一次模型评估。通过这种方式，可以快速验证模型的初始性能和确认所有配置（如数据加载、模型初始化、优化器设置等）是否正确无误。，并确保训练过程按预期开始。
    # 建议值：False
    # 理由：除非你需要初始评估结果，否则保持默认值。
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


# Initialize the argument parser and load the script arguments.
# 解析命令行参数并将其转换为数据类实例。
parser = HfArgumentParser(ScriptArguments)
# 获取解析后的第一个数据类实例
script_args = parser.parse_args_into_dataclasses()[0]
# Set the random seed for reproducibility.
set_seed(script_args.seed)
# Load the human stack-exchange-paired dataset for tuning the reward model.
train_dataset = load_dataset(
    "lvwerra/stack-exchange-paired", data_dir="data/reward", split="train", verification_mode="no_checks"
)
# If a training subset size is specified, select the corresponding subset.
if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))
# Load the evaluation dataset.
eval_dataset = load_dataset(
    "lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train", verification_mode="no_checks"
)
# If an evaluation subset size is specified, select the corresponding subset.
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
# 提取模型名称的最后一部分：通过 split("/")[-1] 将 script_args.model_name 按照斜杠分割，取最后一部分。
model_name_split = script_args.model_name.split("/")[-1]
# 生成输出文件名：将提取的模型名称部分与其他参数组合成一个字符串，作为输出文件名。
output_name = (
    f"{model_name_split}_peft_stack-exchange-paired_rmts__{script_args.train_subset}_{script_args.learning_rate}"
)

# 初始化训练参数对象
training_args = TrainingArguments(
    # 设置输出目录
    output_dir=output_name,
    # 设置学习率
    learning_rate=script_args.learning_rate,
    # 设置每个设备的训练批次大小
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    # 设置每个设备的评估批次大小
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    # 设置训练的总轮数
    num_train_epochs=script_args.num_train_epochs,
    # 设置权重衰减率
    weight_decay=script_args.weight_decay,
    # 设置评估策略为按步骤进行
    eval_strategy="steps",
    # 设置评估步骤的间隔
    eval_steps=500,
    # 设置保存策略为按步骤进行
    save_strategy="steps",
    # 设置保存步骤的间隔
    save_steps=500,
    # 设置梯度累积的步骤数
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # 设置是否使用梯度检查点
    gradient_checkpointing=script_args.gradient_checkpointing,
    # 配置Deepspeed以提高训练效率
    deepspeed=script_args.deepspeed,
    # 设置本地排名
    local_rank=script_args.local_rank,
    # 禁止移除未使用的列
    remove_unused_columns=False,
    # 设置标签名称为空列表，表示使用默认值
    label_names=[],
    # 设置是否使用bf16精度训练
    bf16=script_args.bf16,
    # 设置日志记录策略为按步骤进行
    logging_strategy="steps",
    # 设置日志记录步骤的间隔
    logging_steps=10,
    # 设置优化器类型
    optim=script_args.optim,
    # 设置学习率调度器类型
    lr_scheduler_type=script_args.lr_scheduler_type,
    # 设置随机种子以确保结果的可重复性
    seed=script_args.seed,
)


# 加载value-head model and分词器tokenizer.
# 根据script_args中的参数选择tokenizer的名称，如果没有指定tokenizer名称，则使用model_name作为tokenizer名称
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
# 使用选定的tokenizer名称从预训练模型中加载tokenizer，如果需要认证，则使用auth token
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
# 将tokenizer的pad_token设置为eos_token，以确保在需要padding时使用正确的token
tokenizer.pad_token = tokenizer.eos_token


# 创建Lora配置实例
# - task_type: 任务类型，这里设置为序列分类
# - inference_mode: 模型是否处于推理模式，这里设为False，表示模型处于训练模式
# - r: Lora机制的秩，影响模型的大小和性能
# - lora_alpha: Lora机制的缩放因子，用于调整权重的重要性
# - lora_dropout: 在Lora层中应用的dropout率，用于防止过拟合
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# 初始化模型架构和权重
# script_args.model_name 指定了预训练模型的名称
# num_labels=1 表示分类任务只有一个标签
# torch_dtype=torch.bfloat16 设置模型权重的精度为半精度浮点数，以优化性能和资源消耗
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
)
# 拿到微调后的模型
# get_peft_model 是一个自定义函数，根据peft_config配置模型的微调参数
# 这一步是为了在保持模型大部分参数不变的情况下，通过少量参数的调整来适配新任务
model = get_peft_model(model, peft_config)
# 打印模型的可训练参数，以便确认微调设置是否正确应用
# 这一步很重要，因为它帮助开发者确认模型中哪些部分在训练时会被更新
model.print_trainable_parameters()

# 设置其他参数如padding token, use_cache, num_proc, original_columns
# Need to do this for gpt2, because it doesn't have an official pad token.
# Set the padding token to be the same as the end-of-sentence token.
tokenizer.pad_token = tokenizer.eos_token
# Set the padding token ID in the model configuration to be the same as the end-of-sentence token ID.
model.config.pad_token_id = tokenizer.eos_token_id
# Configure the model to use or not use caching based on the gradient checkpointing setting.
model.config.use_cache = not script_args.gradient_checkpointing
# Define the number of processing cores to be used, which can be increased if more processors are available.
num_proc = 24
# Store the original column names of the training dataset for subsequent processing or reference.
original_columns = train_dataset.column_names


# 预处理数据集
# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    """
    预处理函数，用于将输入的示例数据转换为模型所需的格式。
    主要功能是将问题和对应的两个回答分别进行分词处理，并保存相应的输入ID和注意力掩码。
    参数:
    examples (dict): 包含 "question", "response_j", "response_k" 等键的字典，对应于一批问题和两个可能的回答。
    返回:
    new_examples (dict): 包含 "input_ids_j", "attention_mask_j", "input_ids_k", "attention_mask_k" 等键的字典，分别存储分词后的输入ID和注意力掩码。
    """
    # 初始化一个新的字典来存储分词后的输入ID和注意力掩码
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }

    # 遍历每个问题及其对应的两个回答
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        # 对第一个回答进行分词处理, 设置 truncation=True 确保文本不会过长。
        # 得到的返回值tokenized_j 是一个字典，包含以下键值对：
        # input_ids：表示输入文本的 token ID 列表。每个 token ID 对应于分词器词汇表中的一个词。
        # attention_mask：表示注意力掩码的列表，用于指示哪些位置是有效的 token（通常用 1 表示有效，0 表示填充或无效部分）。用于帮助模型区分实际内容和填充内容。在批处理中，不同长度的序列会被填充到相同的长度，注意力掩码确保模型只关注实际内容。
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        # 对第二个回答进行分词处理
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)
        # 将分词后的输入ID和注意力掩码分别添加到新的字典中
        # tokenized_j["input_ids"] 是对第一个回答进行分词处理后得到的输入ID列表list。其中包含的是整数，这些整数代表输入文本经过分词器处理后得到的 token ID。
        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
    # 返回包含分词后信息的新字典
    return new_examples



# 预处理和过滤数据集preprocess the dataset
train_dataset = train_dataset.map(
    # 调用preprocess_function 函数对数据集进行批量处理。
    preprocess_function,
    # 设置 batched=True 表示按批次处理数据。
    batched=True,
    # 使用 num_proc 参数指定并行处理的进程数。
    num_proc=num_proc,
    # 移除数据集中指定的原始列。
    remove_columns=original_columns,
)
# 使用 filter 方法对 train_dataset 进行筛选。筛选条件是通过一个 lambda 函数定义的，检查 input_ids_j 和 input_ids_k 的长度是否都小于等于 script_args.max_length。
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length,
    num_proc=num_proc,
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length,
    num_proc=num_proc,
)


# 数据整理，填充对齐长度
# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    """
    用于奖励模型的数据整理器，它在整理数据时会进行填充操作。
    这个数据整理器主要用于处理奖励模型的输入数据。它会将一批数据中的所有特征填充到相同的长度，
    以便能够进行批量处理。这个类继承自dataclass，因此它的属性在实例化后是不可变的。
    属性:
        tokenizer (PreTrainedTokenizerBase): 用于对输入数据进行编码的分词器。
        padding (Union[bool, str, PaddingStrategy]): 控制如何进行填充的策略。可以是布尔值、字符串或PaddingStrategy对象。
        pad_to_multiple_of (Optional[int]): 如果指定了这个值，那么填充后的序列长度将会是这个值的倍数。
        return_tensors (str): 指定返回的张量类型。通常是'pt'（PyTorch张量）。
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        当类的实例被调用时执行的方法。
        这个方法的主要工作是整理输入特征，并将它们分成两个批次（batch_j和batch_k），
        然后对这两个批次的数据进行填充操作，最后返回一个包含所有填充后特征的字典。
        参数: features (list[dict[str, Any]]): 一个包含所有特征的列表，每个特征都是一个字典。  
        返回: dict[str, Any]: 一个包含填充后特征的字典，包括两个输入序列的input_ids和attention_mask，以及一个表示是否返回损失的标志。
        """
        # 分别初始化两个列表，用于存储两种不同的特征
        features_j = []
        features_k = []
        # 遍历输入的特征列表，将它们分别添加到对应的列表中
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        # 使用分词器的pad方法对特征进行填充，生成batch_j和batch_k
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # 构建最终的返回字典，包括两种输入序列的input_ids和attention_mask，以及是否返回损失的标志
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# 初始化一个评估指标对象，以便后续用于模型性能评估。Define the metric that we'll use for validation. 
accuracy = evaluate.load("accuracy")


# 定义计算模型预测准确率的函数
def compute_metrics(eval_pred):
    """
    Computes the accuracy metric for the model's predictions.
    This function calculates the accuracy of the model by comparing the predicted values (rewards_j and rewards_k)
    with the actual labels. Here, it is assumed that the correct situation is when rewards_j is greater than rewards_k.
    The purpose of this comparison is to verify how often the model's predictions align with this assumption.
    Parameters:
    - eval_pred: A tuple containing the model's predicted values and labels (predictions, labels).
    Returns:
    - The accuracy of the model's predictions, calculated by comparing the predicted values with the labels.
    """
    # Extract the model's predicted values and labels from eval_pred
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    # Use argmax to determine which column has the higher value (j or k) for each row, resulting in an array of 0s and 1s
    predictions = np.argmax(predictions, axis=0)
    # 创建一个与predictions相同形状的数组，用于存储标签值，初始化为0，因为原始数据回答j的preference大于回答k的preference
    labels = np.zeros(predictions.shape)
    # Use the accuracy.compute method to calculate the accuracy of the model's predictions
    return accuracy.compute(predictions=predictions, references=labels)


# 定义RewardTrainer奖励模型训练类
class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://huggingface.co/papers/2203.02155
    # 定义计算奖励模型损失loss的方法
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for reward training using the InstructGPT pairwise logloss method.
        Parameters:
        - model: The model used for prediction.
        - inputs: A dictionary containing two sets of input data (input_ids_j, attention_mask_j) and (input_ids_k, attention_mask_k).
        - return_outputs: Whether to return the output in addition to the loss. Default is False.
        Returns:
        - If return_outputs is True, returns the loss and a dictionary containing the rewards for both sets of inputs.
        - If return_outputs is False, returns only the loss.
        """
        # Calculate the reward for the first set of inputs
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        # Calculate the reward for the second set of inputs
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        # Compute the loss using the pairwise logloss formula
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        # If set to return outputs, return the loss and rewards
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        # Otherwise, return only the loss
        return loss


# 进行训练模型Train the model, woohoo.
# 初始化RewardTrainer对象用于模型的训练和评估
# 参数model: 指定需要训练的模型
# 参数args: 包含训练过程中的各种配置和超参数
# 参数train_dataset: 训练数据集，用于模型训练
# 参数eval_dataset: 评估数据集，用于模型评估
# 参数compute_metrics: 指定评估模型性能的指标计算函数
# 参数data_collator: 数据整理器，用于在训练和评估前对数据进行必要的处理
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

# 训练很早期阶段进行评估
# 如果命令行参数中包含了评估第一步的请求，则定义一个回调类以实现此功能
if script_args.eval_first_step:
    # EvaluateFirstStepCallback类继承自TrainerCallback，用于在训练的第一步结束后触发评估
    class EvaluateFirstStepCallback(TrainerCallback):
        # 当训练步骤结束时调用on_step_end方法
        def on_step_end(self, args, state, control, **kwargs):
            # 如果当前是全局训练的第一步，则设置should_evaluate为True，以触发评估流程
            if state.global_step == 1:
                control.should_evaluate = True
    # 向trainer添加EvaluateFirstStepCallback回调，使其能够在训练过程中发挥作用
    trainer.add_callback(EvaluateFirstStepCallback())

# Resume training from a checkpoint if specified
trainer.train(script_args.resume_from_checkpoint)
# Save the last checkpoint of the model for future use or evaluation
print("Saving last checkpoint of the model")
model.save_pretrained(output_name + "_peft_last_checkpoint")
