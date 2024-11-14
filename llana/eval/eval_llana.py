import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
from llana.conversation import conv_templates, SeparatorStyle
from llana.utils import disable_torch_init
from llana.model import *
from llana.model.utils import KeywordsStoppingCriteria
from llana.data import ObjectNeRFDataset_Eval
from llana.data.utils import DataCollatorForNeRFTextDataset_Eval
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
from llana.train.train_llana import DataArguments_Eval
from llana import conversation as conversation_lib

import json
import glob

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLaNA.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float16).to(args.device)
    model.initialize_tokenizer_nf2vec_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    return model, tokenizer, conv

def load_dataset(data_path, data_folder, anno_folder, hst_dataset, text_data, tokenizer, data_args):

    dataset = ObjectNeRFDataset_Eval(
        split='test',
        root = data_path,
        data_folder = data_folder,
        anno_folder = anno_folder,
        hst_dataset = hst_dataset,
        conversation_type=text_data,
        tokenizer=tokenizer,  # * load vec data
        data_args=data_args)

    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, vecs, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            vecs=vecs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation_caption_qa(model, tokenizer, conv, dataloader, annos, output_file_path):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    responses = []
    annos_iter = iter(annos)
    for batch in tqdm(dataloader):
        object_id = batch["object_id"]
        vecs = batch["vecs"].to(model.device).to(model.dtype)   # * tensor of B, N, C(3)
        input_ids = batch["input_ids"].to(model.device)         # * tensor of B, L
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        outputs = generate_outputs(model, tokenizer, input_ids, vecs, stopping_criteria) # List of str, length is B

        output_iter = iter(outputs)
        
        for output in outputs:
            anno_dict = next(annos_iter)
            obj_id = anno_dict["object_id"]
            convs = anno_dict["conversations"]
            for i in range(0, len(convs), 2):
                if convs[i]["from"]=="human":
                    q = convs[i]["value"].replace("<point>\n", "").strip()
                else:
                    raise Exception('The first conversation should be from human.')
                if convs[i+1]["from"]=="gpt":
                    gt_a = convs[i+1]["value"].strip()
                else:
                    raise Exception('The second conversation should be from gpt.')
                responses.append({
                    "object_id": obj_id,
                    "question": q,
                    "ground_truth": gt_a,
                    "model_output": output
                })
                print('\nquestion:', q)
                print('ground_truth: ', gt_a)
                print('result: ', output)
                print('= = = = = = = = = = = = = = = = = = = = = = = = = = =')
    

    # save the results to a JSON file
    with open(output_file_path, 'w') as fp:
        json.dump(responses, fp, indent=2)

    # * print info
    print(f"Saved results to {output_file_path}")

    return responses


def main(args):
    output_folder = os.path.join(args.output_dir, args.model_name.split('/')[-2], args.model_name.split('/')[-1])
    os.makedirs(output_folder, exist_ok=True)
    if args.hst_dataset:
        output_filename = f"hst.json"
    else:
        output_filename = f"{args.text_data}_Shapenet.json"
    
    output_file_path = os.path.join(output_folder, output_filename)
    args.device = torch.device(f'cuda:{args.device}')

    if os.path.exists(output_file_path):
        print(f'[INFO] {args.output_file_path} already exists.')
    else:
        if args.hst_dataset:
            with open(os.path.join(args.data_path, 'hst.json'), 'r') as fp:
                annos = json.load(fp)
        else:
            if args.text_data=="brief_description":
                with open(os.path.join(args.data_path, 'test', args.anno_folder, 'conversations_brief.json'), 'r') as fp:
                    brief_annos = json.load(fp)
            if  args.text_data == "detailed_description" or args.text_data == "single_round" or args.text_data == "multi_round":
                with open(os.path.join(args.data_path, 'test', args.anno_folder, 'conversations_complex.json'), 'r') as fp:
                    complex_annos = json.load(fp)
    
            # parse json to get desired data
            annos = []
            if args.text_data == "brief_description":
                annos += brief_annos
            else:   # single_round or multi_round
                round_data = [anno for anno in complex_annos if anno['conversation_type'] == args.text_data]
                annos += round_data

        model, tokenizer, conv = init_model(args)
        nf2vec_config = model.get_model().nf2vec_config
        parser = transformers.HfArgumentParser((DataArguments_Eval))
        data_args = parser.parse_args_into_dataclasses()[0]
        data_args.point_token_len = nf2vec_config['point_token_len']
        data_args.mm_use_point_start_end = nf2vec_config['mm_use_point_start_end']
        data_args.nf2vec_config = nf2vec_config
        
        dataset = load_dataset(args.data_path, args.data_folder, args.anno_folder, args.hst_dataset, args.text_data, tokenizer, data_args)
        data_collator = DataCollatorForNeRFTextDataset_Eval(tokenizer=tokenizer)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers, data_collator)
        
        print(f'[INFO] Start generating results for {output_file_path}.')
        results = start_generation_caption_qa(model, tokenizer, conv, dataloader, annos, output_file_path)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="", required=True, help="Name of the model to evaluate.") 
    parser.add_argument("--device", type=int, default=0, help="idx of the GPU to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save the evaluation results.")

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/shapenerf_text", required=False)
    parser.add_argument("--data_folder", default="vecs", type=str, help="Name of folder with embeddings.")
    parser.add_argument("--anno_folder", default="texts", type=str, help="Name of folder with conversations.")
    parser.add_argument("--text_data", type=str, default="brief_description", choices=["brief_description", "detailed_description", "single_round", "multi_round"], required=False)
    parser.add_argument("--hst_dataset", action="store_true", help="if True, the HST dataset is used")
    
    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    args = parser.parse_args()

    main(args)