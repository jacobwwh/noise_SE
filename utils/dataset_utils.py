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

class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, args, classified_data):
        self.examples = []
        for js_list in classified_data:
            for js in js_list:
                self.examples.append(convert_examples_to_features_noisyclassification(js,tokenizer,args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.examples[i].original_label is not None:
            return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),torch.tensor(self.examples[i].original_label)
        else:
            return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
