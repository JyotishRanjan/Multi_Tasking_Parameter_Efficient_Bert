from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch


def dataLoader(base_model_name,batch_size = 16):
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)
    
    # create tokenizer function
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', return_tensors='pt')

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    
    train_dataset = train_dataset.remove_columns(['split', 'genre', 'dataset', 'year', 'sid'])
    test_dataset = test_dataset.remove_columns(['split', 'genre', 'dataset', 'year', 'sid'])
    val_dataset = val_dataset.remove_columns(['split', 'genre', 'dataset', 'year', 'sid'])

    # Convert your dataset to DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([x['input_ids'] for x in batch])
        attention_mask = torch.tensor([x['attention_mask'] for x in batch])
        token_type_ids = torch.tensor([x['token_type_ids'] for x in batch])
        labels = torch.tensor([x['score'] for x in batch], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
        

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader,val_loader,test_loader


