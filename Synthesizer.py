import torch
import torch.nn as nn
from transformers import (BertTokenizer, BertModel, BertConfig,
                          BertForSequenceClassification,
                          PretrainedConfig, PreTrainedModel,
                          TrainingArguments, TrainerCallback, Trainer)


class Synthesizer(PreTrainedModel):
    config_class = BertConfig
    
    def __init__(self, max_program_len, max_data_len, config):
        super().__init__(config)
        self.nn_model = BertModel(config)
        
        self.max_program_len = max_program_len
        self.max_data_len = max_data_len
        self.sections = (max_data_len, max_data_len, max_program_len)
        
        # project original output to allow for program prediction
        self.proj_instruction = nn.Linear(config.hidden_size, 2 * max_data_len + max_program_len)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing = .05)
        
    def forward(self, input_ids, attention_mask = None, label_A = None, label_B = None, label_C = None, **kwargs):
        # the pooled output is just the first token in the sequence (which should be the cls token)
        # this can be seen at https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
        # by text searching for "bertpooler"
        pooled_output = self.nn_model(input_ids, attention_mask).pooler_output
        all_logits = self.proj_instruction(pooled_output)
        logits_A, logits_B, logits_C = all_logits.split(self.sections, dim = -1)
        
        loss = None
        if None not in (label_A, label_B, label_C):
            loss = self.loss_func(logits_A, label_A.view(-1))
            loss += self.loss_func(logits_B, label_B.view(-1))
            loss += self.loss_func(logits_C, label_C.view(-1))

        return {"loss": loss, 'labels': (label_A, label_B, label_C),
                'logits': (logits_A, logits_B, logits_C)}


# helper function to create special tokens
def resize_embedding(synthesizer_model, new_size = 30000):
    prev_embed = synthesizer_model.nn_model.embeddings.word_embeddings.weight.data
    old_size = prev_embed.shape[0]
    
    new_embed = nn.Embedding(new_size, synthesizer_model.config.hidden_size)
    
    # match variance of original embedding
    prev_std = prev_embed.std(dim = -1).mean().item()
    new_std = new_embed.weight.std(dim = -1).mean().item()
    new_embed.weight.data /= (new_std / prev_std)
    
    new_embed.weight.data[:old_size, :] = prev_embed
    
    synthesizer_model.nn_model.embeddings.word_embeddings = new_embed


def compute_accuracy(eval_prediction):
    label_A, label_B, label_C = eval_prediction.predictions[0]
    logits_A, logits_B, logits_C = eval_prediction.predictions[1]
    
    acc_A = (logits_A.argmax(axis = 1) == label_A.reshape(-1)).mean()
    acc_B = (logits_B.argmax(axis = 1) == label_B.reshape(-1)).mean()
    acc_C = (logits_C.argmax(axis = 1) == label_C.reshape(-1)).mean()

    acc = (acc_A + acc_B + acc_C) / 3
    return {'accuracy': acc}