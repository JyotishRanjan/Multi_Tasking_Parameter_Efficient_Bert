from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def char_to_token_index(context, char_start, char_end, tokenizer):
    encoding = tokenizer(context, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = encoding['offset_mapping']
    token_start, token_end = None, None

    for idx, (start, end) in enumerate(offsets):
        if start <= char_start < end:
            token_start = idx
        if start < char_end <= end:
            token_end = idx
            break
    
    if token_end is None or token_start is None:
        return -1, -1

    return token_start, token_end

def dataLoader(base_model_name, batch_size=16):
    dataset = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)

    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['context'], examples['question'],
                                       truncation=True, padding='max_length',
                                       max_length=512, return_offsets_mapping=True)
        token_starts = []
        token_ends = []

        for i in range(len(examples['context'])):
            char_start = examples['answers'][i]['answer_start'][0]
            char_end = char_start + len(examples['answers'][i]['text'][0])
            token_start, token_end = char_to_token_index(examples['context'][i], char_start, char_end, tokenizer)
            if token_start == -1 or token_end == -1:
                token_start, token_end = 0, 0  # Setting default values to prevent errors
            token_starts.append(token_start)
            token_ends.append(token_end)

        tokenized_examples['token_start'] = token_starts
        tokenized_examples['token_end'] = token_ends

        return tokenized_examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]

    def collate_fn(batch):
        input_ids = torch.tensor([x['input_ids'] for x in batch], dtype=torch.long)
        attention_mask = torch.tensor([x['attention_mask'] for x in batch], dtype=torch.long)
        token_starts = torch.tensor([x['token_start'] for x in batch], dtype=torch.long)
        token_ends = torch.tensor([x['token_end'] for x in batch], dtype=torch.long)

        token_start = torch.zeros(size=(len(batch), 512), dtype=torch.float)
        token_end = torch.zeros(size=(len(batch), 512), dtype=torch.float)

        for i in range(len(batch)):
            token_start[i][token_starts[i]] = 1
            token_end[i][token_ends[i]] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_start": token_start,
            "answer_end": token_end
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader
