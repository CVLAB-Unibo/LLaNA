import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Sequence
from llana import conversation as conversation_lib

IGNORE_INDEX = -100


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1_eval(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        # add "USER" followed by the question
        question = source[0]
        role = roles[question["from"]]
        assert role == conv.roles[0], f"{i}"
        conv.append_message(role, question["value"]) 
        # add "ASSISTANT" followed by \n, without any answer
        answer = source[1]
        role = roles[answer["from"]]
        assert role == conv.roles[1], f"{i}"
        conv.append_message(role, '') 
        
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal_point_cloud(  # adds placeholder tokens for the sequence of tokens from the Point Encoder: <point_patch><point_patch>...
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    for source in sources:
        for sentence in source:
            replace_token = default_point_patch_token * point_token_len     # point_token_len=513, default_point_patch_token='<point_patch>'
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    return sources

@dataclass
class DataCollatorForNeRFTextDataset(object):
    """Collate examples for mixed dataset with text and nf2vec data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'vecs' in instances[0]:
            vecs = [instance['vecs'] for instance in instances]
            if all(x is not None and x.shape == vecs[0].shape for x in vecs):
                batch['vecs'] = torch.stack(vecs)
            else:
                batch['vecs'] = vecs # * return as lists

        return batch

@dataclass
class DataCollatorForNeRFTextDataset_Eval(object):
    """Collate examples for mixed dataset with text and nf2vec data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, object_id = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "object_id"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            object_id = object_id
        )

        if 'vecs' in instances[0]:
            vecs = [instance['vecs'] for instance in instances]
            if all(x is not None and x.shape == vecs[0].shape for x in vecs):
                batch['vecs'] = torch.stack(vecs)
            else:
                batch['vecs'] = vecs # * return as lists

        return batch