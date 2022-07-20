from .utils import (
    get_args,
    set_seed,
)
from transformers.data.processors.utils import InputFeatures
from .tuner import Tuner


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    is_tf_dataset = False

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = [
            tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length,),
            tokenizer.encode_plus(example.text_b, add_special_tokens=True, max_length=max_length,),
        ]
        input_ids = [inputs[i]["input_ids"] for i in range(len(inputs))]
        token_type_ids = [inputs[i]["token_type_ids"] for i in range(len(inputs))]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [[1 if mask_padding_with_zero else 0] * len(input_ids[i]) for i in range(len(input_ids))]

        # assert len(input_ids[0]) == len(input_ids[1]), f"Error with input pair length {len(input_ids[0])} {len(input_ids[1])}"

        # Zero-pad up to the sequence length.
        padding_length = [max_length - len(input_ids[i]) for i in range(len(input_ids))]
        input_ids = [input_ids[i] + ([pad_token] * padding_length[i]) for i in range(len(input_ids))]
        attention_mask = [attention_mask[i] + ([0 if mask_padding_with_zero else 1] * padding_length[i]) for i in range(len(attention_mask))]
        token_type_ids = [token_type_ids[i] + ([pad_token_segment_id] * padding_length[i]) for i in range(len(token_type_ids))]

        assert len(input_ids[0]) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(input_ids[1]) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask[0]) == max_length, f"Error with input length {len(attention_mask)} vs {max_length}"
        assert len(attention_mask[1]) == max_length, f"Error with input length {len(attention_mask)} vs {max_length}"
        assert len(token_type_ids[0]) == max_length, f"Error with input length {len(token_type_ids)} vs {max_length}"
        assert len(token_type_ids[1]) == max_length, f"Error with input length {len(token_type_ids)} vs {max_length}"

        if output_mode == "classification":
            # TODO: fix `Found dtype Long but expected Float` error and remove float()
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features

def main():
    args = get_args()
    # Set seed
    set_seed(args)
    finetuner = Tuner(args)
    finetuner.prepare(args)
    train_dataset = finetuner.load_and_cache_examples(
        args,
        args.task_name,
        evaluate=False,
        convert_examples_to_features=convert_examples_to_features
    )
    global_step, tr_loss = finetuner.train(
        args,
        train_dataset,
        convert_examples_to_features=convert_examples_to_features
    )

if __name__ == "__main__":
    main()
