import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
PREFIX = 'translate English to SQL: '

# "5pm" / "7 pm" / "1145am" / "7:30 pm" -> the 24-hour integer the database uses (1700 / 1900 / 1145 / 1930)
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}|\d{1,4})\s?([ap])\.?m\.?(?![a-z])")
_NOON_RE = re.compile(r"\b(?:12\s)?noon\b")

def normalize_times(text):
    def repl(m):
        digits = m.group(1).replace(':', '')
        d = int(digits)
        h, minute = (d, 0) if len(digits) <= 2 else (d // 100, d % 100)
        if m.group(2) == 'p' and h < 12:
            h += 12
        elif m.group(2) == 'a' and h == 12:
            h = 0
        return str(h * 100 + minute)
    return _NOON_RE.sub('1200', _TIME_RE.sub(repl, text))

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, normalize_times_flag=False):
        '''
        Data processing for the T5 model.
          * encoder input : "translate English to SQL: " + question, tokenized with the t5-small tokenizer
          * decoder target: the SQL query tokens + </s>      (not available for the test split)
          * decoder input : [pad] + target[:-1]   (T5 starts decoding from the pad token, id 0; this is the
                            right-shifted target used for teacher forcing)
        '''
        self.split = split
        self.normalize_times_flag = normalize_times_flag
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl = load_lines(os.path.join(data_folder, f'{split}.nl'))
        if self.normalize_times_flag:
            nl = [normalize_times(q) for q in nl]
        nl = [PREFIX + q for q in nl]
        enc = tokenizer(nl, truncation=True, max_length=512)['input_ids']

        if split == 'test':
            # no SQL for the test set; each item is just the encoder ids
            return [{'enc': e} for e in enc]

        sql = load_lines(os.path.join(data_folder, f'{split}.sql'))
        # T5's vocabulary has no '<' (it becomes <unk>), so write it as a word and undo it after generation
        sql = [q.replace('<', 'less_than') for q in sql]
        dec = tokenizer(sql, truncation=True, max_length=1024)['input_ids']   # ends with </s>
        return [{'enc': e, 'dec_in': [PAD_IDX] + d[:-1], 'dec_tgt': d} for e, d in zip(enc, dec)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([torch.tensor(x['enc']) for x in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([torch.tensor(x['dec_in']) for x in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([torch.tensor(x['dec_tgt']) for x in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([torch.tensor(x['enc']) for x in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, normalize_times_flag=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, normalize_times_flag)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, normalize_times_flag=False):
    train_loader = get_dataloader(batch_size, "train", normalize_times_flag)
    dev_loader = get_dataloader(test_batch_size, "dev", normalize_times_flag)
    test_loader = get_dataloader(test_batch_size, "test", normalize_times_flag)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x