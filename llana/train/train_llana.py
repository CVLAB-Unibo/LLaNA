# Adopted from https://github.com/OpenRobotLab/PointLLM/. Below is the original copyright:
#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
#   Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List
import os
import torch

import transformers
from llana.train.llana_trainer import LLaNATrainer

from llana import conversation as conversation_lib
from llana.model import *
from llana.data import make_object_nerf_data_module

# * logger
from llana.utils import build_logger

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")

@dataclass
class DataArguments:
    root: str = field(default=None, metadata={"help": "Root of dataset"})
    data_folder: str = field(default=None, metadata={"help": "Name of the folder containing vecs from nf2vec."})
    anno_folder: str = field(default=None, metadata={"help": "Name of the folder containing the conversations on the 3D data."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    conversation_types: List[str] = field(default_factory=lambda: ["brief_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True

@dataclass
class DataArguments_Eval:
    device: int = field(default=0)
    hst_dataset: bool = field(default=False)
    text_data: str = field(default="brief_description")
    img_view: str = field(default="first")
    model_name: str = field(default="llava_vicuna-13b")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."}) # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)

    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning
    stage_2: bool = field(default=False) # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None) # * path to the pre-trained projector & output_embed & input_embed
    detatch_point_token: bool = field(default=False) # * deprecated

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info" # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')
    
    if training_args.model_debug:
        # * do not load checkpoint, load from config
        config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        model = LLaNA._from_config(config)
    else:
        model = LLaNA.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path
        )

    model.config.use_cache = False

    if training_args.fix_llm:
        # * This will fix all the parameters
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().vec_proj.requires_grad_(True) 
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    
    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_config
        logger.info("embedding projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("embedding projection layer is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
        model.initialize_tokenizer_nf2vec_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
    else:
        # * stage2
        model.initialize_tokenizer_nf2vec_config_wo_embedding(tokenizer=tokenizer)
    
    nf2vec_config = model.get_model().nf2vec_config

    data_args.point_token_len = nf2vec_config['point_token_len']
    data_args.mm_use_point_start_end = nf2vec_config['mm_use_point_start_end']
    data_args.nf2vec_config = nf2vec_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_nerf_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)

    trainer = LLaNATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
