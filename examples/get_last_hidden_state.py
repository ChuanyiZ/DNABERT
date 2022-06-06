import torch
from transformers import BertModel, BertConfig, DNATokenizer
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dir_to_pretrained_model = "../model/6-new-12w-0"
path_to_config = 'https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json'
# test_file = "../../data/gene_APC.txt"
# output_png = "test_APC.pdf"
test_file = "sample_data/ft/6/dev.tsv"
output_png = "test_dev_1001.png"

# dir_to_pretrained_model = "../../data/pretrain6/"
# path_to_config = '../../data/pretrain6/config.json'
# test_file = "../../data/coronavirus_leader_body.txt"
# output_png = "test_coronavirus.pdf"

config = BertConfig.from_pretrained(path_to_config)
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

seqs = []
labels = []
with open(test_file) as ifile:
    next(ifile)
    cnt = 0
    for line in ifile:
        cnt += 1
        sequence, label = line.strip().split('\t')
        seqs.append(sequence)
        labels.append(label)

model_input = []
for idx, seq in enumerate(seqs):
    model_input.append(tokenizer.encode_plus(seq,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True)["input_ids"])
model_input = torch.tensor(model_input, dtype=torch.long)
# model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one

batch_size = 8

pooled_encodings = []
for i in range(0, model_input.shape[0], batch_size):
    output = model(model_input[i:(i+batch_size), :])
    pooled_encodings.append(output[1].detach().numpy())

pooled_encodings = np.vstack(pooled_encodings)
labels = np.array(labels)
print(pooled_encodings)

lde = umap.UMAP().fit_transform(pooled_encodings)
for x in np.unique(labels):
    plt.scatter(
        lde[labels==x, 0],
        lde[labels==x, 1],
        s=5,
        alpha=0.3,
        label=x)
plt.legend()
plt.gca().set_aspect('equal', 'datalim')
plt.savefig(output_png)
