import gzip
import json
import re
from io import StringIO
import torch
import tokenize
import math
from transformers import PLBartTokenizer
import random
class Data_Preprocessor:
    def __init__(self,
                 tokenizer,
                 args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_len = args.block_size

    def inp2features(self, instance):
        inp = instance['code'].replace('\n', '').split()
        tgt = instance['comment'].replace('\n','').split()
        inp_ids = self.tokenize(inp, max_len = 512)
        tgt_ids = self.tokenize(tgt, max_len=100)
        if inp_ids == None or tgt_ids == None:
            return None, None
        return inp_ids, tgt_ids


    def tokenize(self, code, max_len = None):
        max_len = max_len if max_len is not None else self.max_len
        toks = []
        source = []
        for i, tok in enumerate(code):
            if self.args.model_type == 't5':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)
            if len(sub_tks) == 0:
                continue
            toks += sub_tks
            source += self.tokenizer.convert_tokens_to_ids(sub_tks)
            if len(source) >= max_len:
                #print('out of length ', i, ' / ', len(code))
                break
        if len(source) >= max_len - 2:
            source = source[:max_len - 2]
            source = [self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
        else:
            source = [self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
            pad_len = max_len - len(source)
            source += pad_len * [self.tokenizer.pad_token_id]
        return torch.tensor(source).long()

def main():
    langs = ['java']

    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")

    data_pre = Data_Preprocessor(tokenizer)
    parsers = {}
    for lang in langs:
        LANGUAGE = Language('../../code/data_preprocess/my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    inp_data_path = './data/small/train.buggy-fixed.buggy'
    tgt_data_path = './data/java_test.jsonl.gz'
    instances = load_jsonl_gz(tgt_data_path)
    # with open(inp_data_path, 'r') as f:
    #     inp_lines = f.readlines()
    # with open(tgt_data_path, 'r') as f:
    #     tgt_lines = f.readlines()
    #assert(len(inp_lines) == len(tgt_lines))

    for instance in instances:
        inp, tgt = data_pre.inp2features(instance, parsers, 'java', 'vul')
        print('inp is ', inp)
        print('inp is ', data_pre.tokenizer.convert_ids_to_tokens(inp))
        print('tgt is ', data_pre.tokenizer.convert_ids_to_tokens(tgt))
    # for i in range(len(inp_lines)):
    #     data_pre.inp2features((inp_lines[i], tgt_lines[i]), parsers, 'java')



if __name__ == "__main__":
    main()