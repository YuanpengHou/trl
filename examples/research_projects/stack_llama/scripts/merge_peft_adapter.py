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
# 导入PeftConfig和PeftModel类，用于处理微调相关的配置和模型
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          HfArgumentParser)

"""
该代码的主要功能是合并一个基础模型Base Model和一个适配器模型Adapter Model，并将合并后的模型保存到指定路径，同时推送到 Hugging Face Hub。
"""

@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the merged model.
    """
    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


# 创建一个HfArgumentParser实例，用于解析命令行参数
parser = HfArgumentParser(ScriptArguments)
# 解析命令行参数，并将结果转换为ScriptArguments数据类实例
script_args = parser.parse_args_into_dataclasses()[0]
# 确保提供了适配器模型的名称
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
# 确保提供了基础模型的名称
assert script_args.base_model_name is not None, "please provide the name of the Base model"
# 确保提供了合并后模型的输出名称
assert script_args.output_name is not None, "please provide the output name of the merged model"

# Load the configuration of the fine-tuning model from the specified adapter model name
peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
# Determine the model type based on the task type in the fine-tuning configuration
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the pretrained model from the specified path or name
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
# Set the model to evaluation mode, ready for inference or evaluation tasks
model.eval()

# 合并模型并卸载，此方法用于在模型训练和评估过程中进行合并，并在合并后卸载模型
model = model.merge_and_unload()

# 将模型保存到指定输出路径
model.save_pretrained(f"{script_args.output_name}")
# 将分词器保存到指定输出路径
tokenizer.save_pretrained(f"{script_args.output_name}")
# 将模型推送到Hub，使用指定的输出名称，不使用临时目录
model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
