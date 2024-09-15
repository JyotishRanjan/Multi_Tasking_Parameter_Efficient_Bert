import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import evaluate 

accuracy = evaluate.load("accuracy")

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patience = 5

save_directory = '/home/jyotish/isro/MYProjects/model_checkpoints'

# Corrected train function
def train(model, train_loader, val_loader, loss_fn, num_epochs=3, learning_rate=5e-5, task='task1'):
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
            start_positions = batch['answer_start'].to(device)
            end_positions = batch['answer_end'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            start_logits, end_logits = outputs['start_logits'], outputs['end_logits']
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = start_loss + end_loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        cur_loss, _ = validate(model, val_loader, loss_fn, task)
        
        if cur_loss < best_loss:
            best_model = model
            best_loss = cur_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(best_model.state_dict(), os.path.join(save_directory, 'task3_best_model.pt'))
            print(f"New best model saved with validation loss: {cur_loss:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement for {patience} consecutive epochs.")
            break
        
    return best_model



def validate(model, val_loader, loss_fn, task='task1'):
    model.eval()
    total_val_loss = 0
    val_y_start = []
    val_y_end = []
    val_pred_start = []
    val_pred_end = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['answer_start'].to(device)
            end_positions = batch['answer_end'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            start_logits, end_logits = outputs['start_logits'], outputs['end_logits']
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = start_loss + end_loss
            total_val_loss += loss.item()
            
            val_pred_start.extend(start_logits.argmax(dim=-1).cpu().numpy())
            val_pred_end.extend(end_logits.argmax(dim=-1).cpu().numpy())
            val_y_start.extend(start_positions.argmax(dim=-1).cpu().numpy())
            val_y_end.extend(end_positions.argmax(dim=-1).cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    accuracy_start = accuracy.compute(predictions=val_pred_start, references=val_y_start)
    accuracy_end = accuracy.compute(predictions=val_pred_end, references=val_y_end)

    print(f'Validation Loss: {avg_val_loss:.4f} Start Accuracy: {accuracy_start["accuracy"]:.4f} End Accuracy: {accuracy_end["accuracy"]:.4f}')
    
    return avg_val_loss, (val_pred_start, val_pred_end)
