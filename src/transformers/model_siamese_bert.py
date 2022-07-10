"""Siamese BERT model. """


import logging
# import math
# import os

# import torch
from torch import cat, nn, abs, sum
from torch.nn import CrossEntropyLoss, MSELoss, CosineSimilarity
# import torch.autograd as autograd

# from .activations import gelu, gelu_new, swish
# from .configuration_bert import BertConfig
# from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
# from .modeling_utils import PreTrainedModel, prune_linear_layer
from .modeling_bert import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)

class SiameseBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        is_freeze=False
    ):
        if is_freeze:
            for name, param in self.bert.named_parameters():
                if name.startswith("bert.encoder.layer.11") or name.startswith("bert.pooler"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        outputs0 = self.bert(
            input_ids[:, 0, :],
            attention_mask=attention_mask[:, 0, :],
            # position_ids=position_ids[:, 0, :],
            # head_mask=head_mask[:, 0, :],
            # inputs_embeds=inputs_embeds[:, 0, :],
        )
        outputs1 = self.bert(
            input_ids[:, 1, :],
            attention_mask=attention_mask[:, 1, :],
            # position_ids=position_ids[:, 1, :],
            # head_mask=head_mask[:, 1, :],
            # inputs_embeds=inputs_embeds[:, 1, :],
        )
        
        pooled_output0 = outputs0[1]
        pooled_output1 = outputs1[1]

        # pooled_output0 = self.dropout(pooled_output0)
        # pooled_output1 = self.dropout(pooled_output1)
        # logits = self.classifier()
        diff = abs(pooled_output0 - pooled_output1)
        tri = cat((pooled_output0, pooled_output1, diff), dim=1)
        logits = self.classifier(tri)

        # TODO: get rid of `+ outputs0[2:]`
        outputs = (logits,) + outputs0[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
