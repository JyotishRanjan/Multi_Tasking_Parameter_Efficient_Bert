import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from peft import LoraConfig
import torch.nn.functional as F


class BertForSEQCLF(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(BertForSEQCLF, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output):
        logits = self.classifier(sequence_output[:, 0])  # Take the [CLS] token's hidden state
        return logits

class BertForTextSummarization(nn.Module):
    def __init__(self, hidden_size):
        super(BertForTextSummarization, self).__init__()
        self.decoder = nn.Linear(hidden_size, hidden_size)  # You may want to use a more sophisticated decoder

    def forward(self, sequence_output):
        return self.decoder(sequence_output)
    
import torch.nn.functional as F

class BertForSTS(nn.Module):
    def __init__(self, hidden_size):
        super(BertForSTS, self).__init__()
        self.dense = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, pooled_output):
        # pooled_output = sequence_output[:, 0]  # Using [CLS] token output
        logits = self.dense(pooled_output)
        # scaled_logit = 5 * self.sigmoid(logits)
         # Approximate sigmoid using two ReLUs
        approx_sigmoid = F.relu(logits) - F.relu(logits - 5)
        return approx_sigmoid
        # return scaled_logit
        
class BertForQuestionAnswering(nn.Module):
    def __init__(self, hidden_size):
        super(BertForQuestionAnswering, self).__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits' :start_logits, "end_logits" : end_logits}


class UnifiedModel(nn.Module):
    def __init__(self, base_model_name):
        super(UnifiedModel, self).__init__()
        config = BertConfig.from_pretrained(base_model_name)
        self.base_model = BertModel.from_pretrained(base_model_name, config=config)
        
        task1_lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=['query', 'value']
        )
        task2_lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=['query', 'value']
        )
        task3_lora_config = LoraConfig(
            task_type="SEQ2SEQ_LM",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=['query', 'value']
        )
        
        self.base_model.add_adapter(task1_lora_config, adapter_name="adapter_task1")
        self.base_model.add_adapter(task2_lora_config, adapter_name="adapter_task2")
        self.base_model.add_adapter(task3_lora_config, adapter_name="adapter_task3")
        
        self.task1_head = BertForSEQCLF(self.base_model.config.hidden_size, 2)
        self.task2_head = BertForSTS(self.base_model.config.hidden_size)
        self.task3_head = BertForQuestionAnswering(self.base_model.config.hidden_size)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids = None, task="task1"):
        
        if task == "task1":
            self.base_model.set_adapter("adapter_task1")
            base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
            sequence_output = base_outputs[0]
            return self.task1_head(sequence_output)
        elif task == "task2":
            self.base_model.set_adapter("adapter_task2")
            base_outputs = self.base_model(input_ids, attention_mask=attention_mask,token_type_ids = token_type_ids)
            sequence_output = base_outputs.pooler_output
            return self.task2_head(sequence_output)
        elif task == "task3":
            self.base_model.set_adapter("adapter_task3")
            base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
            sequence_output = base_outputs[0]
            return self.task3_head(sequence_output)
        else:
            raise ValueError(f"Unknown task: {task}")

        

        
        
def Model(base_model_name = "bert-base-uncased",num_labels_task1 = 2,num_labels_task3 = 2):
    unified_model = UnifiedModel(base_model_name)
    return unified_model

# class BertForSEQCLF(nn.Module):
#     def __init__(self, hidden_size, num_labels):
#         super(BertForSEQCLF, self).__init__()
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, sequence_output):
#         logits = self.classifier(sequence_output[:, 0, :])  # Take the [CLS] token's hidden state
#         return logits

# class BertForTextSummarization(nn.Module):
#     def __init__(self, hidden_size):
#         super(BertForTextSummarization, self).__init__()
#         self.decoder = nn.Linear(hidden_size, hidden_size)  # You may want to use a more sophisticated decoder

#     def forward(self, sequence_output):
#         return self.decoder(sequence_output)
    
# class BertForSTS(nn.Module):
#     def __init__(self, hidden_size):
#         super(BertForSTS, self).__init__()
#         self.dense = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, sequence_output):
#         pooled_output = sequence_output[:, 0]  # Using [CLS] token output
#         logits = self.dense(pooled_output)
#         return self.sigmoid(logits)


# class UnifiedModel(nn.Module):
#     def __init__(self, base_model_name, num_labels_task1, num_labels_task3):
#         super(UnifiedModel, self).__init__()
#         config = BertConfig.from_pretrained(base_model_name)
#         self.base_model = BertModel.from_pretrained(base_model_name, config=config)
        
#         task1_lora_config = LoraConfig(
#             task_type="SEQ_CLS",
#             r=4,
#             lora_alpha=32,
#             lora_dropout=0.01,
#             target_modules=['query', 'value']
#         )
#         task2_lora_config = LoraConfig(
#             task_type="SEQ_CLS",
#             r=4,
#             lora_alpha=32,
#             lora_dropout=0.01,
#             target_modules=['query', 'value']
#         )
#         task3_lora_config = LoraConfig(
#             task_type="SEQ_CLS",
#             r=4,
#             lora_alpha=32,
#             lora_dropout=0.01,
#             target_modules=['query', 'value']
#         )
#         task4_lora_config = LoraConfig(
#             task_type="SEQ2SEQ_LM",
#             r=4,
#             lora_alpha=32,
#             lora_dropout=0.01,
#             target_modules=['query', 'value']
#         )
        
#         self.base_model.add_adapter(task1_lora_config, adapter_name="adapter_task1")
#         self.base_model.add_adapter(task2_lora_config, adapter_name="adapter_task2")
#         self.base_model.add_adapter(task3_lora_config, adapter_name="adapter_task3")
#         self.base_model.add_adapter(task4_lora_config, adapter_name="adapter_task4")
        
#         self.task1_head = BertForSEQCLF(self.base_model.config.hidden_size, num_labels_task1)
#         self.task2_head = BertForSTS(self.base_model.config.hidden_size)
#         self.task3_head = BertForSEQCLF(self.base_model.config.hidden_size, num_labels_task3)
#         self.task4_head = BertForTextSummarization(self.base_model.config.hidden_size)
    
#     def forward(self, input_ids, attention_mask=None, task="task1"):
        
#         if task == "task1":
#             self.base_model.set_adapter("adapter_task1")
#             base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
#             sequence_output = base_outputs[0]
#             return self.task1_head(sequence_output)
#         elif task == "task2":
#             self.base_model.set_adapter("adapter_task2")
#             base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
#             sequence_output = base_outputs[0]
#             return self.task2_head(sequence_output)
#         elif task == "task3":
#             self.base_model.set_adapter("adapter_task3")
#             base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
#             sequence_output = base_outputs[0]
#             return self.task3_head(sequence_output)
#         elif task == "task4":
#             self.base_model.set_adapter("adapter_task4")
#             base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
#             sequence_output = base_outputs[0]
#             return self.task4_head(sequence_output)
#         else:
#             raise ValueError(f"Unknown task: {task}")
        
        
