from tokenizers import SentencePieceBPETokenizer
import torch
from transformers.data.processors.utils import InputFeatures
torch.cuda.empty_cache()
from torch.utils.data import DataLoader, IterableDataset
from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_dna import DNATokenizer
from transformers.model_siamese_bert import (
    SiameseBertTripletLoss,
)
from transformers.configuration_bert import BertConfig
from transformers.data.processors.glue import DnaPairProcessor
from .utils import (
    get_args,
    convert_example_pairs_to_features,
)
from .tuner import Tuner
from .dataset import SentenceLabelDataset

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def main():
    args = get_args()
    # Set seed
    finetuner = Tuner(
        args,
        "classification",
        BertConfig,
        SiameseBertTripletLoss,
        DNATokenizer,
        DnaPairProcessor,
        convert_examples_to_features=convert_example_pairs_to_features
    )
    train_dataset = finetuner.load_and_cache_examples()
    global_step, tr_loss = finetuner.train(train_dataset, SentenceLabelDataset)
    print(f" global_step = {global_step}, average loss = {tr_loss}")


if __name__ == "__main__":
    main()
