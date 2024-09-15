from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

def dataLoader(base_model_name,batch_size = 16):
    dataset = load_dataset('glue','sst2')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)
    
    # create tokenizer function
    def tokenize_fn(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True,return_tensors='pt')
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    
    # Convert your dataset to DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([x['input_ids'] for x in batch])
        attention_mask =  torch.tensor([x['attention_mask'] for x in batch])
        labels = torch.tensor([x['label'] for x in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader,val_loader,test_loader
