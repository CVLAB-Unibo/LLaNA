import argparse
import torch
from torch.utils.data import DataLoader
import os
from llana.conversation import conv_templates, SeparatorStyle
from llana.utils import disable_torch_init
from llana.model import *
from llana.model.utils import KeywordsStoppingCriteria
from llana.data import ObjectNeRFDataset_Eval, ObjectPointCloudDataset, ObjectPointCloudDataset_Eval
from llana.data.utils import DataCollatorForNeRFTextDataset_Eval
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
from llana.eval.evaluator import start_evaluation
import sys
from llana.train.train_nerfllm import DataArguments, DataArguments_Eval
from llana import conversation as conversation_lib


import os
import json
import glob
import re

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail.",
    "<question>"
]

name_to_class_id = {
    "table": "04379243",
    "car": "02958343",
    "chair": "03001627",
    "airplane": "02691156",
    "sofa": "04256520",
    "rifle": "04090263",
    "lamp": "03636649",
    "watercraft": "04530566",
    "bench": "02828884",
    "pistol": "03948459",
    "loudspeaker": "03691459",
    "cabinet": "02933112",
    "display": "03211117",
    "telephone": "04401088",
    "phone": "02992529"
}

class_id_to_name = {v: k for k, v in name_to_class_id.items()}

class_names = list(name_to_class_id)

synonyms = {
    "display": ["TV"],
    "telephone": ["smartphone", "iPhone", "blackberry", "Samsung galaxy"],
    "phone": ["smartphone", "iPhone", "blackberry", "Samsung galaxy"],
    "chair": ["stool"],
    "table": ["desk"],
    "loudspeaker": ["speaker"],
    "watercraft": ["boat", "ship"]
}

def find_class_ids(model_id):
    # Get the root directory
    root_dir = '/media/data2/aamaduzzi/datasets/nerf2vec_renderings_data/data_TRAINED'

    # Use glob to find all directories that end with the model_id
    model_dirs = glob.glob(f'{root_dir}/*/{model_id}')

    # Extract the class_id from each directory
    class_ids = [os.path.basename(os.path.dirname(dir)) for dir in model_dirs]

    return class_ids

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float16).to(args.device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    return model, tokenizer, conv

def load_dataset(pcs_path, anno_root, anno_folder, tokenizer, hst_dataset, text_data, data_args):
    dataset = ObjectPointCloudDataset_Eval(
        pcs_path=pcs_path,
        anno_root=anno_root,
        anno_folder=anno_folder,
        tokenizer=tokenizer,
        hst_dataset=hst_dataset,
        split='test',
        conversation_type=text_data,
        data_args=data_args) # * load point cloud only

    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
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

def start_generation_caption_qa(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    
    qs = PROMPT_LISTS[prompt_index]
    '''
    # prepare annotation file to suit text_data
    if "single_round" in args.text_data or "multi_round" in args.text_data:
        if qs!="<question>":
            raise Exception('For single_round or multi_round, prompt_index should be 3.')
    elif qs != "<question>":
        # * convert annos file to <object_id>: <gt_answer>
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}
    # * convert annos file to <object_id>: <gt_answer>
    if qs == "<question>":
        annos = {anno["object_id"]: anno["conversations"] for anno in annos}'''
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    
    if qs != "<question>":   # get the input prompt offline, since it is the same for all the data
        if mm_use_point_start_end:
            qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
        else:
            qs = default_point_patch_token * point_token_len + '\n' + qs
        
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        
        print('prompt:', prompt)
        inputs = tokenizer([prompt])

        input_ids_ = torch.as_tensor(inputs.input_ids).to(args.device) # * tensor of 1, L

        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []
    annos_iter = iter(annos)
    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].to(model.device).to(model.dtype) # * tensor of B, N, C(3)
        input_ids = batch["input_ids"].to(model.device)         # * tensor of B, L
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B
        
        for output in outputs:
            anno_dict = next(annos_iter)
            obj_id = anno_dict["object_id"]
            convs = anno_dict["conversations"]
            for i in range(0, len(convs), 2):
                if convs[i]["from"]=="human":
                    if qs == "<question>":
                        q = convs[i]["value"].replace("<point>\n", "").strip()
                    else:
                        q = PROMPT_LISTS[prompt_index]
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
    

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(responses, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return responses

def start_generation_classification(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):

    class_question = "Which object is this? Choose one among the following: "
    class_question += ', '.join(class_names) + '. '
    class_question += "Important: Answer with one single word."
    class_question =  "Which object is this?"
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    
    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + class_question
    else:
        qs = default_point_patch_token * point_token_len + '\n' + class_question
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    
    print('prompt:', prompt)
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).to(args.device) # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []
    annos_iter = iter(annos)
    n_correct = 0
    n_total = 0
    for batch in tqdm(dataloader):
        object_id = batch["object_id"]
        point_clouds = batch["point_clouds"].to(model.device).to(model.dtype)   # * tensor of B, N, C(3)
        input_ids = input_ids_
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        output_iter = iter(outputs)
        
        for output in outputs:
            correct = False
            anno_dict = next(annos_iter)
            obj_id = anno_dict["object_id"]
            
            # extract gt class
            gt_class_ids = find_class_ids(obj_id)
            gt_class_names = [class_id_to_name[gt_class_id] for gt_class_id in gt_class_ids]
    
            n_total += len(gt_class_names)
            
            for gt_class_name in gt_class_names:
                if gt_class_name in output or any(synonym in output for synonym in synonyms.get(gt_class_name, [])):
                    print('Correct!')
                    correct = True
                    n_correct +=1
                    print('total correct:', n_correct)
                else:
                    # check if the output is a substring of the gt or vice versa
                    #if any(output_class_name in gt_class_name or gt_class_name in output_class_name for output_class_name in output_class_names for gt_class_name in gt_class_names):
                    #    print('Correct!')
                    #    n_correct += 1
                    #    print('total correct:', n_correct)
                    #else:
                    print('Incorrect!')
            print('total_gt:', n_total)
            print('= = = = = = = = = = = = = = = = = = = = = = = = = = =')
            print('\nobject_id:', obj_id)
            print('output:', output)
            print('ground_truth_class: ', gt_class_names)
            print('correct:', correct)
            responses.append({
                "object_id": obj_id,
                "question": class_question,
                "output": output,
                "ground_truth_class": gt_class_names,
                "correct": correct
            })
    
    print(f'==== CLASSIFICATION ACCURACY: {n_correct/n_total} ====')
    classification_results = {"n_total": n_total, "n_correct": n_correct, "accuracy": n_correct/n_total}
    with open(os.path.join(output_dir, "Shapenet_classification_evaluated.json"), 'w') as fp:
        json.dump(classification_results, fp, indent=2)
    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(responses, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return responses


def main(args):
    # * ouptut
    if 'mesh_from_gt' in args.pcs_path:
        args.output_dir = os.path.join('evaluation_results_from_mesh', os.path.join(args.model_name.split('/')[-1]))
    else:
        args.output_dir = os.path.join('evaluation_results', os.path.join(args.model_name.split('/')[-1]))
    if args.hst_dataset:
        args.output_file = f"hst.json"
    elif args.classification:
        args.output_file = f"Shapenet_classification.json"
    else:
        args.output_file = f"{args.text_data}_Shapenet.json"
    
    args.output_file_path = os.path.join(args.output_dir, args.output_file)
    args.device = torch.device(f'cuda:{args.device}')

    # * annotations
    args.anno_path = os.path.join(args.anno_root, 'test', args.anno_folder)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        if args.hst_dataset:
            anno_file = 'shapenet_text_ds/hst_dataset_filtered_pointllm.json'
            with open(anno_file, 'r') as fp:
                annos = json.load(fp)
        elif args.classification: 
            anno_file = os.path.join(args.anno_path, 'conversations_shapenet_text_brief_FULL_pointllm.json')
            with open(anno_file, 'r') as fp:
                annos = json.load(fp)
        else:
            if args.text_data=="brief_description":
                anno_file = os.path.join(args.anno_path, 'conversations_shapenet_text_brief_FULL_pointllm.json')
                with open(anno_file, 'r') as fp:
                    brief_annos = json.load(fp)
            if  args.text_data == "detailed_description" or args.text_data == "single_round" or args.text_data == "multi_round":
                anno_file = os.path.join(args.anno_path, 'conversations_shapenet_text_complex_FULL_pointllm.json')
                with open(anno_file, 'r') as fp:
                    complex_annos = json.load(fp)

            # parse json to get desired data
            annos = []
            if args.text_data == "brief_description":
                annos += brief_annos
            else:   # single_round or multi_round
                round_data = [anno for anno in complex_annos if anno['conversation_type'] == args.text_data]
                annos += round_data

        model, tokenizer, conv = init_model(args)
        point_backbone_config = model.get_model().point_backbone_config
        parser = transformers.HfArgumentParser((DataArguments_Eval))
        data_args = parser.parse_args_into_dataclasses()[0]
        data_args.point_token_len = point_backbone_config['point_token_len']
        data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
        data_args.point_backbone_config = point_backbone_config
        
        dataset = load_dataset(args.pcs_path, args.anno_root, args.anno_folder, tokenizer, args.hst_dataset, args.text_data, data_args)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
        
        print(f'[INFO] Start generating results for {args.output_file}.')
        if args.classification:
            results = start_generation_classification(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)
        else:
            results = start_generation_caption_qa(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_eval:
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        
        pipeline = pipeline(
        "text-generation",
        model=args.eval_llama_model,
        torch_dtype=torch.float16,  # TODO: test with bfloat16, too
        device_map="auto",
    )
        
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.model_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--device", type=int, default=3, help="idx of the GPU to use")

    # * dataset type
    parser.add_argument("--anno_root", type=str, default="data/llana_data/nerfllm", required=False)
    #parser.add_argument("--pcs_path", type=str, default="/media/data2/aamaduzzi/datasets/mesh_from_nerf/all_test_clouds", required=False)
    parser.add_argument("--pcs_path", type=str, default="/media/data2/aamaduzzi/datasets/mesh_from_gt/all_test_clouds", required=False)
    parser.add_argument("--data_folder", default="shapenet_vecs", type=str, help="Name of folder with embeddings.")
    parser.add_argument("--anno_folder", default="shapenet_texts", type=str, help="Name of folder with conversations.")
    parser.add_argument("--classification", action="store_true", help="if True, the classification task is performed")
    parser.add_argument("--text_data", type=str, default="brief_description", choices=["brief_description", "detailed_description", "single_round", "multi_round"], required=False)
    parser.add_argument("--hst_dataset", action="store_true", help="if True, the HST dataset is used")
    
    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=3)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--eval_llama_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)