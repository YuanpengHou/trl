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

# 导入argparse模块，用于解析命令行参数
import argparse
# 导入os模块，提供与操作系统交互的功能
import os

# 导入Accelerator类，用于加速深度学习模型训练，简化跨GPU/TPU训练和推理过程
from accelerate import Accelerator
# 导入load_dataset函数，用于加载各种格式的数据集
from datasets import load_dataset
# 导入LoraConfig类，用于配置低秩适应（Low-Rank Adaptation）模型参数
from peft import LoraConfig
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入AutoModelForCausalLM类，用于自动加载预训练的因果语言模型
# 导入AutoTokenizer类，用于自动加载预训练的分词器
# 导入TrainingArguments类，用于配置模型训练参数
# 导入logging模块，用于设置日志记录级别和格式
# 导入set_seed函数，用于设置随机种子以确保结果的可重复性
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, logging, set_seed)

# 导入SFTTrainer类，用于训练模型
from trl import SFTTrainer
# 导入ConstantLengthDataset类，用于创建固定长度的数据集
from trl.trainer import ConstantLengthDataset

"""
Fine-Tune Llama-7b on SE paired dataset
"""


def get_args():
    """
    配置并解析命令行参数。
    Returns: Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser()
    
    # 模型路径配置
    parser.add_argument("--model_path", type=str, default="")
    # 数据集名称配置
    parser.add_argument("--dataset_name", type=str, default="lvwerra/stack-exchange-paired")
    # 数据集子集配置
    parser.add_argument("--subset", type=str, default="data/finetune")
    # 数据集分割配置
    parser.add_argument("--split", type=str, default="train")
    # 验证集大小配置
    parser.add_argument("--size_valid_set", type=int, default=4000)
    # 是否使用流式数据加载
    parser.add_argument("--streaming", action="store_true")
    # 数据集洗牌缓冲区大小
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    # 序列长度配置
    parser.add_argument("--seq_length", type=int, default=1024)
    # 最大训练步数配置
    parser.add_argument("--max_steps", type=int, default=10000)
    # 批处理大小配置
    parser.add_argument("--batch_size", type=int, default=4)
    # 梯度累积步数配置
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # 结束标记符ID配置
    parser.add_argument("--eos_token_id", type=int, default=49152)

    # 学习率配置
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # 学习率调度器类型配置
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    # 预热步数配置
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    # 权重衰减配置
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # 本地排名配置
    parser.add_argument("--local_rank", type=int, default=0)
    # 是否使用FP16精度训练
    parser.add_argument("--fp16", action="store_true", default=False)
    # 是否使用BF16精度训练
    parser.add_argument("--bf16", action="store_true", default=False)
    # 是否使用梯度检查点
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    # 随机种子配置
    parser.add_argument("--seed", type=int, default=0)
    # 工作线程数配置
    parser.add_argument("--num_workers", type=int, default=None)
    # 输出目录配置
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    # 日志记录频率配置
    parser.add_argument("--log_freq", default=1, type=int)
    # 评估频率配置
    parser.add_argument("--eval_freq", default=1000, type=int)
    # 保存模型频率配置
    parser.add_argument("--save_freq", default=1000, type=int)

    # 解析并返回命令行参数
    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    Parameters:
    dataset (Dataset): The dataset to be analyzed.
    tokenizer (Tokenizer): The tokenizer used to split text into tokens.
    nb_examples (int): The number of examples to analyze from the dataset.
    Returns:
    float: The average number of characters per token.
    """
    # Initialize counters for characters and tokens
    total_characters, total_tokens = 0, 0
    # Iterate over the specified number of examples in the dataset
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        # Prepare the text of the example
        text = prepare_sample_text(example)
        # Accumulate the number of characters in the text
        total_characters += len(text)
        # Determine the number of tokens based on the type of tokenizer
        if tokenizer.is_fast:
            # For fast tokenizers, get the number of tokens through tokenizer(text).tokens()
            total_tokens += len(tokenizer(text).tokens())
        else:
            # For non-fast tokenizers, get the number of tokens through tokenizer.tokenize(text)
            total_tokens += len(tokenizer.tokenize(text))
    
    # Return the average number of characters per token
    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    This function iterates over all parameters of the given model, calculates the total number of parameters
    and the number of trainable parameters (those that require gradient computation), and prints these numbers
    along with the proportion of trainable parameters.
    Parameters:
    - model: The neural network model, whose parameters are to be counted.
    """
    # Initialize the count of trainable parameters and the total number of parameters to 0
    trainable_params = 0
    all_param = 0
    # Iterate over all parameter tensors in the model
    for _, param in model.named_parameters():
        # Increment the total number of parameters by the number of elements in the current parameter tensor
        all_param += param.numel()
        # If the current parameter tensor requires gradient computation, increment the count of trainable parameters
        if param.requires_grad:
            trainable_params += param.numel()
    # Print the total number of trainable parameters, the total number of parameters, and the proportion of trainable parameters
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset.
    Args:
        example (dict): A sample from the dataset, containing the keys 'question' and 'response_j', with corresponding text values.
    Returns:
        str: A formatted string including the question and answer.
    """
    # Format the question and answer into a single string
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args):
    """
    创建训练和验证数据集。
    该函数根据给定的tokenizer和参数args，加载指定的数据集，将其划分为训练集和验证集，
    并计算数据集的字符与令牌比例。如果启用流式加载，数据集将以流的方式加载，否则将一次性加载。
    参数:
        tokenizer: 用于数据集令牌化的tokenizer。
        args: 包含数据集加载和配置信息的命名空间。
    返回:
        train_dataset: 训练数据集。
        valid_dataset: 验证数据集。
    """
    # 加载数据集，根据args参数配置加载选项
    # - args.dataset_name: 数据集的名称
    # - args.subset: 数据集的子集名称，用于指定数据集的特定版本或类型
    # - args.split: 数据集的分割类型，如"train"、"test"或"validation"
    # - use_auth_token: 是否使用身份验证令牌，用于访问受限制的数据集
    # - args.num_workers: 并行处理数据加载的工作者数量，仅在非流式加载时有效
    # - args.streaming: 是否以流式加载数据集，如果为True，则数据集会被逐行加载，而不是一次性加载到内存中
    # 返回值:
    # - dataset: 加载后的数据集对象，可以根据需要进行迭代和访问
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

    # 根据args.streaming的值决定是否以流式加载数据集, 处理数据集以获取训练和验证数据集
    if args.streaming:
        print("Loading the dataset in streaming mode")
        # 从数据集中取出验证集，大小由args.size_valid_set指定
        valid_data = dataset.take(args.size_valid_set)
        # 跳过验证集，剩余的数据将用作训练集
        train_data = dataset.skip(args.size_valid_set)
        # 对训练数据进行洗牌操作，以增加数据的随机性，buffer_size和seed由参数指定
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        # 如果不使用流式加载，則将数据集按照指定的比例划分为训练集和验证集
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        # 获取划分后的训练集和验证集
        train_data = dataset["train"]
        valid_data = dataset["test"]
    # 打印训练集和验证集的大小信息
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    # 计算数据集的字符与令牌比例
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # 使用ConstantLengthDataset包装训练和验证数据集，初始化一个恒定长度的数据集对象，用于训练
    train_dataset = ConstantLengthDataset(
        tokenizer,  # 用于文本标记化的tokenizer对象
        train_data,  # 训练数据，通常是一个大的文本语料库
        formatting_func=prepare_sample_text,  # 对样本文本进行预处理的函数
        infinite=True,  # 设置数据集为无限模式，意味着数据集会被循环使用
        seq_length=args.seq_length,  # 指定输出序列的长度
        chars_per_token=chars_per_token,  # 每个token对应的字符数，用于估计序列长度
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    # 返回训练和验证数据集
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    """
    运行训练过程。
    参数:
    - args: 包含训练参数的对象。
    - train_data: 训练数据集。
    - val_data: 验证数据集。
    """
    # 打印加载模型的信息
    print("Loading the model")
    # 初始化LoRA（Low-Rank Adaptation）配置对象，用于微调语言模型
    # r: 控制LoRA层中低秩矩阵的秩，影响模型的大小和性能
    # lora_alpha: LoRA层的缩放因子，用于调整模型的输出
    # lora_dropout: 在LoRA层中应用的dropout率，用于防止过拟合
    # bias: 指定是否在LoRA层中使用偏置项，"none"表示不使用
    # task_type: 指定模型的任务类型，这里为"CAUSAL_LM"，即因果语言模型任务
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 设置训练数据集的起始迭代次数为0
    train_data.start_iteration = 0
    # 打印开始主循环的信息
    print("Starting main loop")
    # 初始化训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # 输出目录路径
        dataloader_drop_last=True,  # 是否在数据加载器中丢弃最后一个无法形成完整batch的数据
        eval_strategy="steps",  # 评估策略，按步骤进行评估
        max_steps=args.max_steps,  # 最大训练步数
        eval_steps=args.eval_freq,  # 评估频率，每多少步进行一次评估
        save_steps=args.save_freq,  # 保存频率，每多少步保存一次模型
        logging_steps=args.log_freq,  # 日志记录频率，每多少步记录一次日志
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批次大小
        per_device_eval_batch_size=args.batch_size,  # 每个设备上的评估批次大小
        learning_rate=args.learning_rate,  # 学习率
        lr_scheduler_type=args.lr_scheduler_type,  # 学习率调度器类型
        warmup_steps=args.num_warmup_steps,  # 预热步数
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        gradient_checkpointing=args.gradient_checkpointing,  # 是否使用梯度检查点
        fp16=args.fp16,  # 是否使用16位浮点数训练
        bf16=args.bf16,  # 是否使用脑浮点16位训练
        weight_decay=args.weight_decay,  # 权重衰减率
        run_name="llama-7b-finetuned",  # 运行名称
        report_to="wandb",  # 报告工具，此处使用Weights & Biases
        ddp_find_unused_parameters=False,  # DDP是否查找未使用的参数
    )

    # 从预训练模型路径加载用于因果语言模型的自动模型
    # 该模型以8位精度加载，并根据当前进程的索引映射到设备
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    # 初始化SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )
    # 打印模型的可训练参数
    print_trainable_parameters(trainer.model)
    # 打印训练开始的信息
    print("Training...")
    # 开始训练模型
    trainer.train()
    # 打印保存模型最后检查点的信息
    print("Saving last checkpoint of the model")
    # 保存训练好的模型
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    """
    主函数，用于执行模型的训练和评估流程。
    参数:
    - args: 包含模型路径和其他配置参数的命名空间。
    此函数首先根据提供的模型路径初始化一个分词器，然后创建训练数据集和评估数据集。
    最后，使用这些数据集以及提供的参数运行模型训练过程。
    """
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 创建训练数据集和评估数据集
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    # 运行模型训练
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 确保提供了模型路径，否则抛出异常
    assert args.model_path != "", "Please provide the llama model path"
    # 设置随机种子以保证结果的可重复性
    set_seed(args.seed)
    # 创建输出目录，如果已存在则不操作
    os.makedirs(args.output_dir, exist_ok=True)
    # 设置日志级别为错误，仅显示错误信息
    logging.set_verbosity_error()
    # 调用主函数并传入参数
    main(args)
