from .utils import (
    get_args,
    set_seed,
)
from .tuner import Tuner

def main():
    args = get_args()
    # Set seed
    set_seed(args)
    finetuner = Tuner(args)
    finetuner.prepare(args)
    train_dataset = finetuner.load_and_cache_examples(args, args.task_name, evaluate=False)
    global_step, tr_loss = finetuner.train(args, train_dataset)
    print(f"device: {args.device}\tcheckpoint: {args.model_name_or_path}")

if __name__ == "__main__":
    main()
