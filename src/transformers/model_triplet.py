import logging
from typing import List, Union

import torch
from torch import cat, nn, abs
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

def euclidian_distance(embeddings, squared=False):
    """
    Compute the 2D matrix of euclidian distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances

def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices

def batch_triplet_loss(
    embeddings,
    labels
):
    # Get the pairwise distance matrix
    pairwise_dist = euclidian_distance(embeddings)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss

# Semi-Hard Triplet Loss
# Based on: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py#L71
# Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering: https://arxiv.org/pdf/1503.03832.pdf
def batch_semi_hard_triplet_loss(embeddings, labels, triplet_margin):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
    Returns:
        Label_Sentence_Triplet: scalar tensor containing the triplet loss
    """
    labels = labels.unsqueeze(1)

    pdist_matrix = euclidian_distance(embeddings)

    adjacency = labels == labels.t()
    adjacency_not = ~adjacency

    batch_size = torch.numel(labels)
    pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

    mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(pdist_matrix.t(), [-1, 1]))

    mask_final = torch.reshape(torch.sum(mask, 1, keepdims=True) > 0.0, [batch_size, batch_size])
    mask_final = mask_final.t()

    negatives_outside = torch.reshape(_masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = negatives_outside.t()

    negatives_inside = _masked_maximum(pdist_matrix, adjacency_not)
    negatives_inside = negatives_inside.repeat([1, batch_size])

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = (pdist_matrix - semi_hard_negatives) + triplet_margin

    mask_positives = adjacency.float().to(labels.device) - torch.eye(batch_size, device=labels.device)
    mask_positives = mask_positives.to(labels.device)
    num_positives = torch.sum(mask_positives)

    triplet_loss = torch.sum(torch.max(loss_mat * mask_positives, torch.tensor([0.0], device=labels.device))) / num_positives

    return triplet_loss

def batch_hard_triplet_loss(embeddings, labels, triplet_margin):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
    Returns:
        Label_Sentence_Triplet: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = euclidian_distance(embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + triplet_margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss

def get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct


    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def _masked_minimum(data, mask, dim=1):
    axis_maximums, _ = data.max(dim, keepdims=True)
    masked_minimums = (data - axis_maximums) * mask
    masked_minimums, _ = masked_minimums.min(dim, keepdims=True)
    masked_minimums += axis_maximums

    return masked_minimums

def _masked_maximum(data, mask, dim=1):
    axis_minimums, _ = data.min(dim, keepdims=True)
    masked_maximums = (data - axis_minimums) * mask
    masked_maximums, _ = masked_maximums.max(dim, keepdims=True)
    masked_maximums += axis_minimums

    return masked_maximums

class SiameseBertTripletLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.triplet_margin = 1.0
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.MLP = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
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
        is_freeze=False,
        use_hard=False,
        *args,
    ):
        if is_freeze:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

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
        # tri = self.dropout(tri)
        embeddings = self.MLP(tri)

        if not use_hard:
            # loss = self.batch_triplet_loss(embeddings, labels)
            loss = batch_semi_hard_triplet_loss(embeddings, labels, self.triplet_margin)
        else:
            loss = batch_hard_triplet_loss(embeddings, labels, self.triplet_margin)
        outputs = (loss, embeddings)

        return outputs  # (loss), (hidden_states), (attentions)


class SiameseBertAfterTriplet(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.triplet_margin = 1.0
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.MLP = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, config.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, self.num_labels),
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
            for param in self.MLP.parameters():
                param.requires_grad = True
        elif is_freeze is True:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            for param in self.MLP.parameters():
                param.requires_grad = False
        elif isinstance(is_freeze, str):
            for name, param in self.bert.named_parameters():
                if name.startswith(is_freeze):
                    param.requires_grad = False
            for name, param in self.MLP.named_parameters():
                if name.startswith(is_freeze):
                    param.requires_grad = False
        elif isinstance(is_freeze, list):
            for name, param in self.bert.named_parameters():
                if any(name.startswith(prefix) for prefix in is_freeze):
                    param.requires_grad = False
            for name, param in self.MLP.named_parameters():
                if any(name.startswith(prefix) for prefix in is_freeze):
                    param.requires_grad = False

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
        # tri = self.dropout(cat((pooled_output0, pooled_output1, diff), dim=1))
        embeddings = self.MLP(tri)
        logits = self.classifier(embeddings)

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


class RNNTripletLoss(BertPreTrainedModel):
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
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels, bias=False)
        # self.classifier = nn.Sequential(
        #     nn.Linear(3*config.hidden_size, 3*config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(3*config.hidden_size, 3*config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(3*config.hidden_size, self.config.num_labels),
        # )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        use_hard=False,
    ):
        batch_size = input_ids.shape[0]

        # lstm
        if self.rnn_type == "lstm":
            # zero
            pooled_output = input_ids.view(batch_size, self.split, 3*self.hidden_size)
            _, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            pooled_output = input_ids.view(batch_size, self.split, 3*self.hidden_size)
            _, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        embeddings = ht.squeeze(0).sum(dim=0)
        
        if not use_hard:
            # loss = self.batch_triplet_loss(embeddings, labels)
            loss = batch_semi_hard_triplet_loss(embeddings, labels, self.triplet_margin)
        else:
            loss = batch_hard_triplet_loss(embeddings, labels, self.triplet_margin)
        outputs = (loss, embeddings)

        return outputs  # (loss), logits, (hidden_states), (attentions)