import logging
import os
import datetime
import json
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm, trange
from .utils import (
    MODEL_CLASSES,
    TOKEN_ID_GROUP,
    set_seed,
    _rotate_checkpoints,
)

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
output_modes["siamese"] = "classification"
from transformers import glue_processors as processors
from transformers.data.processors.glue import DnaPairProcessor
processors["siamese"] = DnaPairProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

class Tuner:
    def __init__(self, args) -> None:
        self.device = args.device
        self._setup_logger(args)
    
    def _setup_logger(self, args):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )
        self.logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )

    def prepare(self, args):
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))
        processor = processors[args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        self.config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        self.config.hidden_dropout_prob = args.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        self.config.split = int(args.max_seq_length/512)
        self.config.rnn = args.rnn
        self.config.num_rnn_layer = args.num_rnn_layer
        self.config.rnn_dropout = args.rnn_dropout
        self.config.rnn_hidden = args.rnn_hidden

        self.tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.logger.info('finish loading model')

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(args.device)

        self.logger.info("Training/evaluation parameters %s", args)

    def load_and_cache_examples(
        self,
        args,
        task,
        evaluate=False,
        convert_examples_to_features=convert_examples_to_features
    ) -> TensorDataset:
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
        if args.do_predict:
            cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                str(args.max_seq_length),
                str(task),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = (
                processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            )   

            
            print("finish loading examples")

            # params for convert_examples_to_features
            max_length = args.max_seq_length
            pad_on_left = bool(args.model_type in ["xlnet"])
            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0


            if args.n_process == 1:
                features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id,)
                    
            else:
                n_proc = int(args.n_process)
                if evaluate:
                    n_proc = max(int(n_proc/4),1)
                print("number of processes for converting feature: " + str(n_proc))
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(examples)/n_proc)
                for i in range(1, n_proc+1):
                    if i != n_proc:
                        indexes.append(len_slice*(i))
                    else:
                        indexes.append(len(examples))
            
                results = []
                
                for i in range(n_proc):
                    results.append(p.apply_async(convert_examples_to_features, args=(examples[indexes[i]:indexes[i+1]], self.tokenizer, max_length, None, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
                    print(str(i+1) + ' processor started !')
                
                p.close()
                p.join()

                features = []
                for result in results:
                    features.extend(result.get())
                        

            if args.local_rank in [-1, 0]:
                self.logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def train(
        self,
        args,
        train_dataset,
        convert_examples_to_features=convert_examples_to_features
    ):
        """ Train the model """
        if args.local_rank in [-1, 0] and args.tensorboard:
            summary_dir = os.path.join(args.output_dir, "runs", f"{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}")
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            tb_writer = SummaryWriter(summary_dir)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        self.logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path) and not args.reset_global_step:
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            except:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("  Continuing training from epoch %d", epochs_trained)
            self.logger.info("  Continuing training from global step %d", global_step)
            self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        set_seed(args)  # Added here for reproductibility

        best_auc = 0
        last_auc = 0
        stop_count = 0

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                if step <= warmup_steps:
                    inputs["is_freeze"] = True
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        if (
                            args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(
                                args,
                                convert_examples_to_features=convert_examples_to_features
                            )

                            if args.task_name == "dna690":
                                # record the best auc
                                if results["auc"] > best_auc:
                                    best_auc = results["auc"]

                            if args.early_stop != 0:
                                # record current auc to perform early stop
                                if results["auc"] < last_auc:
                                    stop_count += 1
                                else:
                                    stop_count = 0

                                last_auc = results["auc"]
                                
                                if stop_count == args.early_stop:
                                    self.logger.info("Early stop")
                                    return global_step, tr_loss / global_step


                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        if args.tensorboard:
                            for key, value in logs.items():
                                tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        if args.task_name == "dna690" and results["auc"] < best_auc:
                            continue
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.model.module if hasattr(model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        self.logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        if args.task_name != "dna690":
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            if args.save_optimizer:
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        self.logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.tensorboard:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(
        self,
        args,
        prefix="",
        evaluate=True,
        convert_examples_to_features=convert_examples_to_features
    ):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
        if args.task_name[:3] == "dna" or args.task_name == "siamese":
            softmax = torch.nn.Softmax(dim=1)
            

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(
                args,
                eval_task,
                evaluate=evaluate,
                convert_examples_to_features=convert_examples_to_features
            )

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            #     model = torch.nn.DataParallel(model)

            # Eval!
            self.logger.info("***** Running evaluation {} *****".format(prefix))
            self.logger.info("  Num examples = %d", len(eval_dataset))
            self.logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            probs = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in TOKEN_ID_GROUP else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                    if args.do_ensemble_pred:
                        probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                    else:
                        probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
                elif args.task_name == "siamese":
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
                elif args.task_name == "dnasplice":
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            if args.do_ensemble_pred:
                result = compute_metrics(eval_task, preds, out_label_ids, probs[:,1])
            else:
                result = compute_metrics(eval_task, preds, out_label_ids, probs)
            results.update(result)
            
            if args.task_name == "dna690":
                eval_output_dir = args.result_dir
                if not os.path.exists(args.result_dir): 
                    os.makedirs(args.result_dir)
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "a") as writer:

                if args.task_name[:3] == "dna" or args.task_name == "siamese":
                    eval_result = args.data_dir.split('/')[-1] + " "
                else:
                    eval_result = ""

                self.logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))
                    eval_result = eval_result + str(result[key])[:5] + " "
                writer.write(eval_result + "\n")

        if args.do_ensemble_pred:
            return results, eval_task, preds, out_label_ids, probs
        else:
            return results