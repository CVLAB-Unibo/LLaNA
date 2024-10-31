import argparse
from transformers import AutoTokenizer
import torch
import numpy as np
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
from llana.conversation import conv_templates, SeparatorStyle
from llana.utils import disable_torch_init
from llana.model import *
from llana.model.utils import KeywordsStoppingCriteria

from llana.data import load_objaverse_point_cloud

def load_vec(args):
    object_id = args.object_id
    print(f"[INFO] Loading vec using object_id: {object_id}")
    filename = f"{object_id}.npy"  
    data_root = args.data_path
    data_folder = args.vecs_folder
    data_path = os.path.join(data_root, args.split, data_folder)
    vec = np.load(os.path.join(data_path, filename))
    return object_id, torch.from_numpy(vec).unsqueeze_(0).to(torch.float32)

def init_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_name
    print(f'[INFO] Model name: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LLaNA.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=args.torch_dtype).cuda()
    model.initialize_tokenizer_nf2vec_config_wo_embedding(tokenizer)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    nf2vec_config = model.get_model().nf2vec_config
    
    if mm_use_point_start_end:
        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, nf2vec_config, keywords, mm_use_point_start_end, conv

def start_conversation(args, model, tokenizer, nf2vec_config, keywords, mm_use_point_start_end, conv):
    point_token_len = nf2vec_config['point_token_len']
    default_point_patch_token = nf2vec_config['default_point_patch_token']
    default_point_start_token = nf2vec_config['default_point_start_token']
    default_point_end_token = nf2vec_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation... Enter 'q' to exit the program and enter 'exit' to exit the current conversation.")
    while True:
        print("-" * 80)
    
        # Prompt for data split
        data_split = input("[INFO] Please enter the data split (train, val or test): ")
        
        # Check if the user wants to quit
        if data_split.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Using data split: {data_split}.")
        
        # Prompt for object_id
        object_id = input("[INFO] Please enter the object_id or 'q' to quit: ")
        
        # Check if the user wants to quit
        if object_id.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Chatting with object_id: {object_id}.")

        
        # Update args with new object_id and data_split
        args.object_id = object_id.strip()
        args.split = data_split.strip()
        
        # Load the point cloud data
        try:
            id, vecs = load_vec(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        vecs = vecs.unsqueeze(0).cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(100):
            # This if-else block ensures the initial question from the user is included in the conversation
            qs = input(conv.roles[0] + ': ')
            if qs == 'exit':
                break
            
            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print('prompt:', prompt)
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                print('vecs:', vecs.shape)
                output_ids = model.generate(
                    input_ids,
                    vecs=vecs,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Append the model's response to the conversation history
            conv.pop_last_none_message()
            conv.append_message(conv.roles[1], outputs)
            print(f'{conv.roles[1]}: {outputs}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="outputs/LLaNA_7B_train_stage2_objanerf/slurm_script_31-10-2024_10:23")
    parser.add_argument("--data_path", type=str, default="data/objanerf_text")
    parser.add_argument("--vecs_folder", type=str, default="vecs")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv = init_model(args)
    
    start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv)