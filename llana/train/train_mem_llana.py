# Adopted from https://github.com/OpenRobotLab/PointLLM/. 

import os
import sys

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

# Need to call this before importing transformers.
from llana.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llana.train.train_llana import train

if __name__ == "__main__":
    train()
