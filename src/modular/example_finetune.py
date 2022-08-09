from transformers.configuration_bert import BertConfig
from transformers.data.processors.glue import DnaPairProcessor
from transformers.model_siamese_bert import SiameseBertForSequenceClassification
from transformers.tokenization_dna import DNATokenizer
from .utils import (
    get_args,
    convert_example_pairs_to_features,
)
from .tuner import Tuner

def main():
    args = get_args()
    # Set seed
    finetuner = Tuner(
        args,
        "classification",
        BertConfig,
        SiameseBertForSequenceClassification,
        DNATokenizer,
        DnaPairProcessor,
        convert_examples_to_features=convert_example_pairs_to_features
    )
    train_dataset = finetuner.load_and_cache_examples()
    global_step, tr_loss = finetuner.train(train_dataset)
    print(f" global_step = {global_step}, average loss = {tr_loss}")

if __name__ == "__main__":
    main()
