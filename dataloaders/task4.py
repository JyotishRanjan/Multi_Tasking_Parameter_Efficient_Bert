from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch


def dataLoader(base_model_name,batch_size = 16):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)
    
    # create tokenizer function
    def tokenize_fn(examples):
        inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding='max_length')
        targets = tokenizer(examples['highlights'], max_length=128, truncation=True, padding='max_length')
        inputs['labels'] = targets['input_ids']
        return inputs
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    
    # Convert your dataset to DataLoader
    def collate_fn(batch):
        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [x['attention_mask'] for x in batch]
        labels = torch.tensor([x['labels'] for x in batch])

        # Pad the sequences
        batch_padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        return {
            "input_ids": batch_padded['input_ids'],
            "attention_mask": batch_padded['attention_mask'],
            "labels": labels
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader,val_loader,test_loader
