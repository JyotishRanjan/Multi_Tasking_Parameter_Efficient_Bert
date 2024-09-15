import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import evaluate 

accuracy = evaluate.load("accuracy")

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patience = 5

save_directory = '/home/jyotish/isro/MYProjects/model_checkpoints'

def train(model, train_loader, val_loader,loss_fn, num_epochs=3, learning_rate=5e-5,task = 'task1'):
    best_model = model
    best_loss = float('inf')
        
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            loss = loss_fn(outputs, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        cur_loss,_ = validate(model, val_loader, loss_fn, task)
        
        if cur_loss < best_loss:
            best_model = model
            best_loss = cur_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(best_model.state_dict(), os.path.join(save_directory, 'task1_best_model.pt'))
            print(f"New best model saved with validation loss: {cur_loss:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement for {patience} consecutive epochs.")
            break
        
    return best_model

def validate(model, val_loader, loss_fn, task = 'task1'):
    model.eval()
    total_val_loss = 0
    val_y = []
    val_pred = []
    # device = 'cuda'
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            val_pred = val_pred + list(predictions)
            val_y = val_y + list(labels)

    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f} Accuracy : {accuracy.compute(predictions=val_pred, references=val_y)}')
    
    return avg_val_loss, val_pred

# note try below version if success delete above


# def train(model, train_loader, val_loader,loss_fn, num_epochs=3, learning_rate=5e-5,task = 'task1'):
#     best_model = model
#     best_loss = float('inf')
    
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     total_steps = len(train_loader) * num_epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    

#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss = 0

#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
#             loss = loss_fn(outputs, labels)
#             total_train_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Training Loss: {avg_train_loss:.4f}")

#         cur_loss,_ = validate(model, val_loader, loss_fn, task)
        
#         if(cur_loss < best_loss):
#             best_model = model
#             best_loss = cur_loss
            
#     return best_model

# def validate(model, val_loader, loss_fn, task = 'task1'):
#     model.eval()
#     total_val_loss = 0
#     val_y = []
#     val_pred = []
#     # device = 'cuda'
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Validation"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
#             loss = loss_fn(outputs, labels)
#             total_val_loss += loss.item()

#             predictions = torch.argmax(outputs, dim=-1)
#             val_pred = val_pred + list(predictions)
#             val_y = val_y + list(labels)

#     avg_val_loss = total_val_loss / len(val_loader)
#     print(f'Validation Loss: {avg_val_loss:.4f} Accuracy : {accuracy.compute(predictions=val_pred, references=val_y)}')
    
#     return avg_val_loss,val_pred
