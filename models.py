import transformers
import torch

from typing import *


class BERTmy(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(BERTmy, self).__init__()
        self.rubert = transformers.AutoModel.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence", 
            do_lower_case=True,
            add_additional_tokens=True
        )
        
        hidden_size_output = self.rubert.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_output, hidden_size_output, bias=True),
            torch.nn.Dropout(0.05),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_output, n_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
        token_type_ids: torch.Tensor, output_attentions: bool=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rubert_output = self.rubert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=output_attentions
        )
        if not output_attentions:
            pooled = rubert_output['pooler_output']
        else:
            pooled, attentions = rubert_output['pooler_output'], rubert_output['attentions']

        output = self.classifier(pooled)

        if not output_attentions:
            return output
        else:
            return output, attentions
    
    def configure_optimizer(
        self, use_scheduler: bool,
        bert_lr: float=1e-4, class_lr: float=9e-5,
        scheduler_gamma: float=0.95
    ) -> torch.optim:
        # freeze part of params
        encoder_size = 0
        for param in self.rubert._modules['encoder'].parameters():
            encoder_size += 1
        encoder_size_half = encoder_size // 2
        for idx, param in enumerate(self.rubert._modules['encoder'].parameters()):
            param.requires_grad = False
            if idx >= encoder_size_half:
                break
        
        # Adam
        optimizer = torch.optim.Adam(
            params=[
                {'params':self.rubert._modules['embeddings'].parameters(), 'lr':bert_lr},
                {'params':self.rubert._modules['encoder'].parameters(), 'lr':bert_lr},
                {'params':self.rubert._modules['pooler'].parameters(), 'lr':bert_lr},
                {'params':self.classifier.parameters(), 'lr':class_lr}
            ],
            lr=max(bert_lr, class_lr)
        )
        if use_scheduler:
            # scheduler
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_gamma
            )
        
            return optimizer, scheduler
        
        else:
            return optimizer