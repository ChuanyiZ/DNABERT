"""Siamese BERT model. """


import logging
import math
from typing import List, Union
# import math
# import os

import torch
from torch import cat, nn, abs, sum
from torch.nn import (
    CrossEntropyLoss,
    MSELoss,
    CosineSimilarity,
    TripletMarginLoss,
)
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
        # self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, self.config.num_labels),
        )

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
        is_freeze: Union[bool, str, List[str]] = False,
        *args,
    ):
        if is_freeze is False:
            for param in self.bert.parameters():
                param.requires_grad = True
        elif is_freeze is True:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
        elif isinstance(is_freeze, str):
            for name, param in self.bert.named_parameters():
                if name.startswith(is_freeze):
                    param.requires_grad = False
        elif isinstance(is_freeze, list):
            for name, param in self.bert.named_parameters():
                if any(name.startswith(prefix) for prefix in is_freeze):
                    param.requires_grad = False

        outputs0 = self.bert(
            input_ids[:, 0, :],
            # attention_mask=attention_mask[:, 0, :],
            # position_ids=position_ids[:, 0, :],
            # head_mask=head_mask[:, 0, :],
            # inputs_embeds=inputs_embeds[:, 0, :],
        )
        outputs1 = self.bert(
            input_ids[:, 1, :],
            # attention_mask=attention_mask[:, 1, :],
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
        tri = self.dropout(cat((pooled_output0, pooled_output1, diff), dim=1))
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


class MyBertForLongSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.split = config.split
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden

        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_size,hidden_size=self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

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
        overlap=100,
        max_length_per_seq=500,
        is_freeze=False,
    ):
        # batch_size = input_ids.shape[0]
        # sequence_length = input_ids.shape[1]
        # starts = []
        # start = 0
        # while start + max_length_per_seq <= sequence_length:
        #     starts.append(start)
        #     start += (max_length_per_seq-overlap)
        # last_start = sequence_length-max_length_per_seq
        # if last_start > starts[-1]:
        #     starts.append(last_start)
        

        # new_input_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=input_ids.dtype, device=input_ids.device)
        # new_attention_mask = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=attention_mask.dtype, device=attention_mask.device)
        # new_token_type_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=token_type_ids.dtype, device=token_type_ids.device)

        # for j in range(batch_size):
        #     for i, start in enumerate(starts):
        #         new_input_ids[i] = input_ids[j,start:start+max_length_per_seq]
        #         new_attention_mask[i] = attention_mask[j,start:start+max_length_per_seq]
        #         new_token_type_ids[i] = token_type_ids[j,start:start+max_length_per_seq]

        # if batch_size == 1:
        #     pooled_output = outputs[1].mean(dim=0)
        #     pooled_output = pooled_output.reshape(1, pooled_output.shape[0])
        # else:
        #     pooled_output = torch.zeros([batch_size, outputs[1].shape[1]], dtype=outputs[1].dtype)
        #     for i in range(batch_size):
        #         pooled_output[i] = outputs[1][i*batch_size:(i+1)*batch_size].mean(dim=0)
        
        batch_size = input_ids.shape[0]
        input_ids = input_ids.view(self.split*batch_size, 512)
        attention_mask = attention_mask.view(self.split*batch_size, 512)
        token_type_ids = None
        
        if is_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True
        # outputs = []
        # for i in range(int(math.ceil(self.split * batch_size / 8))):
        #     _, partial_pooled = self.bert(
        #         input_ids[(i * 8):((i+1) * 8), :],
        #         attention_mask=attention_mask[(i * batch_size):((i+1) * batch_size), :],
        #         token_type_ids=token_type_ids,
        #         position_ids=position_ids,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #     )
        #     outputs.append(partial_pooled)
        # outputs = torch.vstack(outputs)
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        # lstm
        if self.rnn_type == "lstm":
            # random
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # c0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # self.hidden = (h0, c0)
            # self.rnn.flatten_parameters()
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # orth
            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # c0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(c0)
            # c0 = autograd.Variable(c0)
            # self.hidden = (h0, c0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # zero
            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, ht = self.rnn(pooled_output)
        else:
            raise ValueError


    
        output = self.dropout(ht.squeeze(0).sum(dim=0))
        logits = self.classifier(output)
        # outputs = (logits,) + outputs[2:]   # add hidden states and attention if they are here

        
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class SiameseBertForLongSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.split = config.split
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden

        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=3*self.hidden_size,hidden_size=3*self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=3*self.hidden_size,hidden_size=3*self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(3*config.hidden_size, 3*config.hidden_size),
            nn.ReLU(),
            nn.Linear(3*config.hidden_size, 3*config.hidden_size),
            nn.ReLU(),
            nn.Linear(3*config.hidden_size, self.config.num_labels),
        )

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
        is_freeze: Union[bool, str, List[str]] = False,
        *args,
    ):
        if is_freeze is False:
            for param in self.bert.parameters():
                param.requires_grad = True
            for param in self.rnn.parameters():
                param.requires_grad = True
        elif is_freeze is True:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            for param in self.rnn.parameters():
                param.requires_grad = False
        elif isinstance(is_freeze, str):
            for name, param in self.bert.named_parameters():
                if name.startswith(is_freeze):
                    param.requires_grad = False
            for name, param in self.rnn.named_parameters():
                if name.startswith(is_freeze):
                    param.requires_grad = False
        elif isinstance(is_freeze, list):
            for name, param in self.bert.named_parameters():
                if any(name.startswith(prefix) for prefix in is_freeze):
                    param.requires_grad = False
            for name, param in self.rnn.named_parameters():
                if any(name.startswith(prefix) for prefix in is_freeze):
                    param.requires_grad = False

        batch_size = input_ids.shape[0]
        input_ids = input_ids.view(self.split*batch_size, 2, 512)
        attention_mask = attention_mask.view(self.split*batch_size, 2, 512)

        if is_freeze:
            for name, param in self.bert.named_parameters():
                if name.startswith("bert"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

        print('{:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))

        with torch.no_grad():
            pooled_output0 = []
            for i in range(int(math.ceil(self.split * batch_size / 128))):
                _, partial_pooled = self.bert(
                    input_ids[(i * 128):((i+1) * 128), 0, :],
                    attention_mask=attention_mask[(i * 128):((i+1) * 128), 0, :],
                    # token_type_ids=token_type_ids,
                    # position_ids=position_ids,
                    # head_mask=head_mask,
                    # inputs_embeds=inputs_embeds,
                )
                pooled_output0.append(partial_pooled)
            pooled_output0 = torch.vstack(pooled_output0)

            pooled_output1 = []
            for i in range(int(math.ceil(self.split * batch_size / 128))):
                _, partial_pooled = self.bert(
                    input_ids[(i * 128):((i+1) * 128), 1, :],
                    attention_mask=attention_mask[(i * 128):((i+1) * 128), 1, :],
                    # token_type_ids=token_type_ids,
                    # position_ids=position_ids,
                    # head_mask=head_mask,
                    # inputs_embeds=inputs_embeds,
                )
                pooled_output1.append(partial_pooled)
            pooled_output1 = torch.vstack(pooled_output1)

        # pooled_output0 = self.dropout(pooled_output0)
        # pooled_output1 = self.dropout(pooled_output1)
        # logits = self.classifier()
        diff = abs(pooled_output0 - pooled_output1)
        tri = cat((pooled_output0, pooled_output1, diff), dim=1)
        # logits = self.classifier(tri)

        # lstm
        if self.rnn_type == "lstm":
            # zero
            pooled_output = tri.view(batch_size, self.split, 3*self.hidden_size)
            _, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            pooled_output = tri.view(batch_size, self.split, 3*self.hidden_size)
            _, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        output = self.dropout(ht.squeeze(0).sum(dim=0))
        logits = self.classifier(output)
        outputs = (logits,)
        # outputs = (logits,) + outputs[2:]   # add hidden states and attention if they are here

        
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class SiameseBertHelper(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
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
        *args,
    ):
        _, pooled_output0 = self.bert(
            input_ids[0, :, :],
        )
        _, pooled_output1 = self.bert(
            input_ids[1, :, :],
        )
        # logits = self.classifier()
        diff = abs(pooled_output0 - pooled_output1)
        tri = cat((pooled_output0, pooled_output1, diff), dim=1)
        return tri

class RNNHelper(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.split = config.split
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden
        self.triplet_margin = 1

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=3*self.hidden_size,hidden_size=3*self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=3*self.hidden_size,hidden_size=3*self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        h0=None,
        c0=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        use_hard=False,
        *args,
    ):
        batch_size = input_ids.shape[0]
        length = input_ids.shape[1] // (3*self.hidden_size)

        # lstm
        if self.rnn_type == "lstm":
            # zero
            pooled_output = input_ids.view(batch_size, length, 3*self.hidden_size)
            if h0 is None or c0 is None:
                _, (ht, ct) = self.rnn(pooled_output)
            else:
                _, (ht, ct) = self.rnn(pooled_output, (h0, c0))
        elif self.rnn_type == "gru":
            pooled_output = input_ids.view(batch_size, length, 3*self.hidden_size)
            _, ht = self.rnn(pooled_output, h0)
        else:
            raise ValueError

        # embeddings = ht.squeeze(0).sum(dim=0)

        return ht, ct # (loss), logits, (hidden_states), (attentions)
