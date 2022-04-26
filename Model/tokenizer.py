import sys
from transformers import  AutoTokenizer

sys.path.append("..")

import argparse
import os


def deepscc_tokenizer(data, max_len=150, pt_model="NTUYG/DeepSCC-RoBERTa"):
    # model_pretained_name = "NTUYG/DeepSCC-RoBERTa"  # 'bert-base-uncased'
    # model_pretained_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pt_model)
    if max_len == 0:
        tokenized = tokenizer.batch_encode_plus(
            data,
            # max_length = max_len,
            pad_to_max_length=True,
            truncation=True
        )
    else:
        tokenized = tokenizer.batch_encode_plus(
            data,
            max_length = max_len,
            pad_to_max_length = True,
            truncation = True
        )

    return tokenized, tokenizer.vocab_size
