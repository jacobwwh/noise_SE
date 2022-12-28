import torch
from torch.utils.data import DataLoader, Dataset

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 original_label=None
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        if original_label is not None:
            self.original_label=original_label
        else:
            self.original_label=None


def convert_examples_to_features_noisyclassification(dicts,tokenizer,args):
    code=' '.join(dicts['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,dicts['label'],dicts['original_label'])
