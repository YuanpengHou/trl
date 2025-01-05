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

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (Adafactor, AutoTokenizer, HfArgumentParser, pipeline,
                          set_seed)

# 导入用于因果语言模型的价值头自动模型、PPO配置和PPO训练器
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
# 导入长度采样器，用于生成特定长度的文本
from trl.core import LengthSampler

# 使用tqdm库的pandas方法初始化进度条
tqdm.pandas()


# 定义个类，用于存储脚本参数
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    # 定义模型名称，默认为空字符串
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    # 定义分词器名称，默认为空字符串
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    # 定义奖励模型名称，默认为空字符串
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    # 定义日志记录工具，如果使用wandb则设置为'wandb'，默认为None
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    # 定义学习率，默认值为1.41e-5
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    # 定义生成输出的最大长度，默认值为128
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    # 定义PPO算法的小批量大小，默认值为1
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    # 定义批处理大小，默认值为32
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    # 定义PPO算法的轮次数量，默认值为4
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    # 定义梯度累积步数，用于在有限内存条件下处理大型数据集
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    # 定义是否使用Adafactor优化器，适用于大规模深度学习模型
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    # 定义是否启用早期停止，以防止过拟合
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    # 定义早期停止的目标KL散度，用于控制模型更新
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    # 定义奖励基线，用于调整奖励值，提高训练效率.
    # 奖励基线 (reward_baseline) 是强化学习中用于调整奖励值的一个参考值。它的主要作用是：
    # 稳定训练过程：通过减去一个基线值，可以减少奖励的方差，从而使训练更加稳定。
    # 加速收敛：适当的奖励基线可以帮助算法更快地收敛到最优策略。
    # 提高效率：通过调整奖励值，可以使模型更专注于有意义的奖励信号，从而提高训练效率。
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    # 定义是否使用批处理文本生成，以提高生成效率
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    # 定义模型保存频率，单位为步骤数
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    # 定义模型保存目录
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    # 定义随机种子，以确保结果的可重复性
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    # 定义训练步骤数，用于控制训练时长
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    # 初始化KL系数，用于适应性和线性控制策略，KL系数惩罚见ppt，KL散度用于衡量新旧策略之间的差异。引入KL惩罚项，可防止模型参数更新过大。
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    # 是否使用适应性KL控制，若否，则使用线性控制
    # 适应性KL（Kullback-Leibler）控制即上行的模型差别的惩罚系数；线性控制是一种简单的固定或按固定规则变化的控制方式。它不依赖于当前的状态或学习进度，而是按照预设的线性规律进行调整。例如，线性控制可以是按固定步长递减学习率，或者按固定的惩罚系数进行约束。
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # 是否以8位精度加载模型，以减少内存使用
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 8bit"})


# 导入HfArgumentParser用于解析命令行参数，并将其转换为ScriptArguments对象
parser = HfArgumentParser(ScriptArguments)
# 解析命令行参数并将其转换为ScriptArguments数据类实例
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
# 从script_args中提取reward_model_name属性，用于后续处理
reward_model_name = script_args.reward_model_name
# 指定要使用的数据集名称，此处为"lvwerra/stack-exchange-paired"
dataset_name = "lvwerra/stack-exchange-paired"
# 初始化PPOConfig配置类，用于设置PPO算法的超参数和训练配置
config = PPOConfig(
    steps=script_args.steps,  # 训练的总步数
    model_name=script_args.model_name,  # 使用的模型名称
    learning_rate=script_args.learning_rate,  # 学习率
    log_with=script_args.log_with,  # 日志记录的方式
    batch_size=script_args.batch_size,  # 批次大小
    mini_batch_size=script_args.mini_batch_size,  # 小批次大小，用于梯度计算
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,  # 梯度累积的步数
    optimize_cuda_cache=True,  # 是否优化CUDA缓存
    early_stopping=script_args.early_stopping,  # 是否启用提前终止训练
    target_kl=script_args.target_kl,  # 目标KL散度
    ppo_epochs=script_args.ppo_epochs,  # PPO算法的轮次
    seed=script_args.seed,  # 随机种子
    init_kl_coef=script_args.init_kl_coef,  # 初始的KL散度系数
    adap_kl_ctrl=script_args.adap_kl_ctrl,  # 是否启用自适应KL散度控制
)

# 加载指定的训练数据集，包括配置参数以禁用数据验证
train_dataset = load_dataset(
    "lvwerra/stack-exchange-paired", data_dir="data/rl", split="train", verification_mode="no_checks"
)
# 从数据集中选择前100000个样本用于训练
train_dataset = train_dataset.select(range(100000))
# 获取并保存原始数据集的列名，以便后续处理或参考
original_columns = train_dataset.column_names


# 定义一个字典，用于配置句子处理的参数
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    # return_all_scores: 表示是否返回所有计算出的分数，而不仅仅是最高分
    "return_all_scores": True,
    # function_to_apply: 指定在分数上应用的函数，"none"表示不应用任何函数
    "function_to_apply": "none",
    # batch_size: 定义处理文本时的批量大小，即一次处理多少个句子
    "batch_size": 16,
    # truncation: 表示是否启用截断，True表示启用，可能会截断过长的句子
    "truncation": True,
}

# 加载指定的分词器
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.
    Args: dataset_name (`str`): The name of the dataset to be loaded.
    Returns: dataloader (`torch.utils.data.DataLoader`): The dataloader for the dataset.
    """
    # 定义使用的处理器数量
    num_proc = 24
    def preprocess_function(examples):
        """
        预处理函数，用于将输入的示例转换为模型所需的格式。
        参数: examples (dict): 包含"question"字段的字典，其中"question"是一个字符串列表。
        返回: dict: 包含"query"和"input_ids"字段的字典，其中"query"是原始问题的格式化版本，"input_ids"是token化后的问题对应的输入ID列表。
        """
        # 初始化新的示例字典
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        # 遍历每个问题，进行格式化和token化
        for question in examples["question"]:
            # 格式化问题
            query = "Question: " + question + "\n\nAnswer: "
            # 使用tokenizer对问题进行token化
            tokenized_question = tokenizer(query, truncation=True)
            # 将格式化后的問題和对应的输入ID添加到新的示例字典中
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples
    # 对训练数据集应用预处理函数
    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    # 过滤掉输入ID长度超过512的示例
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=num_proc)
    # 设置数据集格式为PyTorch张量
    ds.set_format(type="torch")
    # 返回预处理后的数据集
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)

# 定义一个数据处理函数，用于整理数据
def collator(data):
    """
    A function to organize data.
    This function takes a list of dictionaries as input and returns a new dictionary,
    where the keys are the keys from the original dictionaries, and the values are lists 
    containing all the values associated with each key in the input dictionaries.
    Parameters:
    data (list of dict): A list containing multiple dictionaries with the same structure.
    Returns:
    dict: A dictionary where each key corresponds to a list of values, each containing 
          all the values for that key from the input list of dictionaries.
    """
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(config.seed)
# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 加载模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
)

# 选择优化器
optimizer = None
# 如果脚本参数中选择了Adafactor优化器，则创建Adafactor优化器实例
if script_args.adafactor:
    # 使用Adafactor优化器，对模型中所有需要梯度更新的参数进行优化
    # 这里的参数过滤确保了只有requires_grad为True的参数才会被优化
    # Adafactor的参数配置：
    # - scale_parameter: 是否根据参数的大小自适应调整学习率，这里设置为False
    # - relative_step: 是否使用相对步长，这里设置为False
    # - warmup_init: 是否使用预热初始化，这里设置为False
    # - lr: 设置学习率为配置文件中定义的学习率值
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# 初始化PPO（Proximal Policy Optimization）训练器，用PPO算法对模型进行训练
ppo_trainer = PPOTrainer(
    config,  # 配置参数，包含训练过程中的各种设置和超参数
    model,  # 需要训练的模型
    ref_model=None,  # 参考模型，用于计算优势函数，这里设置为None表示不使用外部参考模型
    tokenizer=tokenizer,  # 用于文本数据预处理的分词器
    dataset=dataset,  # 训练数据集，包含所有训练样本
    data_collator=collator,  # 数据整理器，用于将样本批处理成模型所需的格式
    optimizer=optimizer,  # 优化器，用于更新模型参数
)


# Set the device for the PPO trainer
# If using a single process, set the device to 0 for GPU if available, otherwise use CPU
# This is done to avoid a bug in the `pipeline` when multiple processes are not used
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# 初始化奖励模型 Initialize the sentiment analysis pipeline
# This is used to perform sentiment analysis tasks, such as analyzing the sentiment of text data
# The pipeline is configured with the specified reward model, device settings, and tokenizer
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": script_args.load_in_8bit},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

generation_kwargs = {
    # "min_length": -1,  # 最小生成长度，此处注释掉表示未设置最小长度限制
    "top_k": 0.0,  # top-k 采样方法的参数，0表示不使用此方法
    "top_p": 1.0,  # top-p（核）采样方法的参数，1.0表示考虑所有词的概率
    "do_sample": True,  # 是否启用采样，True表示在生成过程中使用概率采样
    "pad_token_id": tokenizer.pad_token_id,  # 填充token的ID，用于补齐序列长度
    "eos_token_id": 100_000,  # 结束token的ID，生成达到此token时停止
}
# 定义生成输出文本的最小长度
output_min_length = 32
# 从脚本参数中获取生成输出文本的最大长度
output_max_length = script_args.output_max_length
# 创建一个长度采样器实例，用于后续生成输出文本的长度
# 这里的LengthSampler假设是一个已定义的类，用于随机选择一个合理的输出长度
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# 使用tqdm包装dataloader以显示训练进度
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # 如果当前epoch超过或等于预设的总训练epochs数，则停止训练
    if epoch >= config.total_ppo_epochs:
        break
    # 从batch中获取问题的tensor表示
    question_tensors = batch["input_ids"]
    # 使用PPO训练器生成响应张量
    # question_tensors: 问题的张量表示
    # return_prompt: 是否返回提示信息，默认为False
    # length_sampler: 输出长度的采样器
    # **generation_kwargs: 其他生成参数，以关键字参数的形式传入
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    # 将生成的响应张量解码为人类可读的文本，skip_special_tokens: 是否跳过特殊标记，如[PAD], [UNK]等
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    # Combine questions and responses for sentiment analysis to compute reward scores,，用于输入给奖励模型得到打分reward score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # 用奖励模型进行对ppo模型的输出结果进行打分 Perform sentiment analysis on the combined text
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # 将reward分数减去基准分数 Compute rewards based on the sentiment analysis results, adjusting for the reward baseline
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    # Perform a PPO training step with the generated questions, responses, and reward scores
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    # Log training statistics
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model periodically
    # Save the model at specified intervals during training
    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
