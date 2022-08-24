import logging
import os
import datetime
import json
import random
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm, trange
from transformers.configuration_utils import PretrainedConfig
from transformers.data.metrics import acc_f1_mcc_auc_pre_rec
from transformers.modeling_bert import BertPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from .utils import (
    MODEL_CLASSES,
    TOKEN_ID_GROUP,
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
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors import DataProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

class Tuner:
    def __init__(self,
        args,
        output_mode: str,
        config_class: PretrainedConfig,
        model_class: BertPreTrainedModel,
        tokenizer_class: PreTrainedTokenizer,
        processor_class: DataProcessor,
        convert_examples_to_features=convert_examples_to_features
    ) -> None:
        self.option = args

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        self.output_mode = output_mode
        self.config_class = config_class
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.processor = processor_class()
        self.convert_examples_to_features = convert_examples_to_features
        self._set_seed(self.option)  # Added here for reproductibility
        self._setup_logger(args)
        self._prepare(args)

    def _set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    
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
            self.device,
            self.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )

    def _prepare(self, args):
        # if args.task_name not in processors:
        #     raise ValueError("Task not found: %s" % (args.task_name))
        label_list = self.processor.get_labels()
        num_labels = len(label_list)

        # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        self.config = self.config_class.from_pretrained(
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

        self.tokenizer = self.tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.logger.info('finish loading model')

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(self.device)

        self.logger.info("Training/evaluation parameters %s", args)

    def load_and_cache_examples(self, evaluate=False) -> TensorDataset:
        if self.option.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.option.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, self.option.model_name_or_path.split("/"))).pop(),
                str(self.option.max_seq_length),
                str(self.option.task_name),
            ),
        )
        if self.option.do_predict:
            cached_features_file = os.path.join(
            self.option.data_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                str(self.option.max_seq_length),
                str(self.option.task_name),
            ),
        )
        if os.path.exists(cached_features_file) and not self.option.overwrite_cache:
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at %s", self.option.data_dir)
            label_list = self.processor.get_labels()
            if self.option.task_name in ["mnli", "mnli-mm"] and self.option.model_type in ["roberta", "xlmroberta"]:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = (
                self.processor.get_dev_examples(self.option.data_dir) if evaluate else self.processor.get_train_examples(self.option.data_dir)
            )   

            
            print("finish loading examples")

            # params for convert_examples_to_features
            max_length = self.option.max_seq_length
            pad_on_left = bool(self.option.model_type in ["xlnet"])
            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
            pad_token_segment_id = 4 if self.option.model_type in ["xlnet"] else 0


            if self.option.n_process == 1:
                features = self.convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=self.output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id,)
                    
            else:
                n_proc = int(self.option.n_process)
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
                    results.append(p.apply_async(self.convert_examples_to_features, args=(examples[indexes[i]:indexes[i+1]], self.tokenizer, max_length, None, label_list, self.output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
                    print(str(i+1) + ' processor started !')
                
                p.close()
                p.join()

                features = []
                for result in results:
                    features.extend(result.get())
                        

            if self.option.local_rank in [-1, 0]:
                self.logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.option.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def train(
        self,
        train_dataset=None,
        sampler=None,
        dataloader=None,
    ):
        """Train the model. Only one of the input arguments (train_dataset, sampler, dataloader) can be used.

        Args:
            train_dataset (optional): Train dataset. Defaults to None.
            sampler (optional): Custom sampler. Defaults to None.
            dataloader (optional): Custom dataloader. Defaults to None.

        Returns:
            global_step, tr_loss / global_step
        """
        if self.option.local_rank in [-1, 0] and self.option.tensorboard:
            summary_dir = os.path.join(self.option.output_dir, "runs", f"{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}")
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            tb_writer = SummaryWriter(summary_dir)

        self.option.train_batch_size = self.option.per_gpu_train_batch_size * max(1, self.n_gpu)
        if train_dataset is not None and sampler is None and dataloader is None:
            train_sampler = RandomSampler(train_dataset) if self.option.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.option.train_batch_size)
        elif train_dataset is not None and sampler is not None and dataloader is None:
            train_data_sampler = sampler(train_dataset)
            train_dataloader = DataLoader(train_data_sampler, batch_size=self.option.train_batch_size, drop_last=True)
        elif train_dataset is None and sampler is None and dataloader is not None:
            train_dataloader = dataloader
        else:
            raise ValueError("Only one of the input arguments (train_dataset, sampler, dataloader) can be used.")

        if self.option.max_steps > 0:
            t_total = self.option.max_steps
            self.option.num_train_epochs = self.option.max_steps // (len(train_dataloader) // self.option.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.option.gradient_accumulation_steps * self.option.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.option.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        warmup_steps = self.option.warmup_steps if self.option.warmup_percent == 0 else int(self.option.warmup_percent*t_total)
        freeze_steps = self.option.freeze_steps if self.option.freeze_percent == 0 else int(self.option.freeze_percent*t_total)
        if self.option.freeze_steps or self.option.freeze_percent:
            is_freeze = True
        elif self.option.freeze_layers is not None:
            is_freeze = self.option.freeze_layers
        else:
            is_freeze = False
        if self.option.semi_hard_percent == 0:
            if self.option.semi_hard_steps > 0:
                semi_hard_steps = self.option.semi_hard_steps
            else:
                semi_hard_steps = None
        else:
            semi_hard_steps = int(self.option.semi_hard_percent*t_total)

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.option.learning_rate, eps=self.option.adam_epsilon, betas=(self.option.beta1,self.option.beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.option.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.option.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.option.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.option.model_name_or_path, "scheduler.pt")))

        if self.option.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.option.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.option.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.option.local_rank], output_device=self.option.local_rank, find_unused_parameters=True,
            )

        # Train!
        self.logger.info("***** Running training *****")
        if train_dataset is not None:
            self.logger.info("  Num examples = %d", len(train_dataset))
        elif dataloader is not None:
            self.logger.info("  Num examples = %d", len(dataloader))
        self.logger.info("  Num Epochs = %d", self.option.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.option.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.option.train_batch_size
            * self.option.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.option.local_rank != -1 else 1),
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.option.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.option.model_name_or_path) and not self.option.reset_global_step:
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(self.option.model_name_or_path.split("-")[-1].split("/")[0])
            except:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // self.option.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.option.gradient_accumulation_steps)

            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("  Continuing training from epoch %d", epochs_trained)
            self.logger.info("  Continuing training from global step %d", global_step)
            self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.option.num_train_epochs), desc="Epoch", disable=self.option.local_rank not in [-1, 0],
        )

        best_auc = 0
        last_auc = 0
        stop_count = 0

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.option.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                if len(batch) == 4:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    inputs["token_type_ids"] = (
                        batch[2] if self.option.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                elif len(batch) == 3:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                if semi_hard_steps is not None and step > semi_hard_steps:
                    inputs["use_hard"] = True
                if step <= freeze_steps:
                    inputs["is_freeze"] = is_freeze
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.option.gradient_accumulation_steps > 1:
                    loss = loss / self.option.gradient_accumulation_steps

                if self.option.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.option.gradient_accumulation_steps == 0:
                    if self.option.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.option.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.option.local_rank in [-1, 0] and self.option.logging_steps > 0 and global_step % self.option.logging_steps == 0:
                        logs = {}
                        if (
                            self.option.local_rank == -1 and self.option.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate()

                            if self.option.task_name == "dna690":
                                # record the best auc
                                if results["auc"] > best_auc:
                                    best_auc = results["auc"]

                            if self.option.early_stop != 0:
                                # record current auc to perform early stop
                                if results["auc"] < last_auc:
                                    stop_count += 1
                                else:
                                    stop_count = 0

                                last_auc = results["auc"]
                                
                                if stop_count == self.option.early_stop:
                                    self.logger.info("Early stop")
                                    return global_step, tr_loss / global_step


                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / self.option.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        if self.option.tensorboard:
                            for key, value in logs.items():
                                tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if self.option.local_rank in [-1, 0] and self.option.save_steps > 0 and global_step % self.option.save_steps == 0:
                        if self.option.task_name == "dna690" and results["auc"] < best_auc:
                            continue
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        output_dir = os.path.join(self.option.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        self.logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(self.option, checkpoint_prefix)

                        if self.option.task_name != "dna690":
                            torch.save(self.option, os.path.join(output_dir, "training_args.bin"))
                            if self.option.save_optimizer:
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        self.logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.option.max_steps > 0 and global_step > self.option.max_steps:
                    epoch_iterator.close()
                    break
            if self.option.max_steps > 0 and global_step > self.option.max_steps:
                train_iterator.close()
                break

        if self.option.local_rank in [-1, 0] and self.option.tensorboard:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(
        self,
        prefix="",
        evaluate=True,
    ):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("mnli", "mnli-mm") if self.option.task_name == "mnli" else (self.option.task_name,)
        eval_outputs_dirs = (self.option.output_dir, self.option.output_dir + "-MM") if self.option.task_name == "mnli" else (self.option.output_dir,)
        if self.option.task_name[:3] == "dna" or self.option.task_name == "siamese":
            softmax = torch.nn.Softmax(dim=1)
            

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(evaluate=evaluate)

            if not os.path.exists(eval_output_dir) and self.option.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            self.option.eval_batch_size = self.option.per_gpu_eval_batch_size * max(1, self.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.option.eval_batch_size)

            # multi-gpu eval
            # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            #     model = torch.nn.DataParallel(model)

            # Eval!
            self.logger.info("***** Running evaluation {} *****".format(prefix))
            self.logger.info("  Num examples = %d", len(eval_dataset))
            self.logger.info("  Batch size = %d", self.option.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            probs = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if self.option.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if self.option.model_type in TOKEN_ID_GROUP else None
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
            if self.output_mode == "classification":
                if self.option.task_name[:3] == "dna" and self.option.task_name != "dnasplice":
                    if self.option.do_ensemble_pred:
                        probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                    else:
                        probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
                elif self.option.task_name == "siamese":
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
                elif self.option.task_name == "dnasplice":
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                preds = np.argmax(preds, axis=1)
            elif self.output_mode == "regression":
                preds = np.squeeze(preds)
            if self.option.do_ensemble_pred:
                result = acc_f1_mcc_auc_pre_rec(preds, out_label_ids, probs[:,1])
            else:
                result = acc_f1_mcc_auc_pre_rec(preds, out_label_ids, probs)
            results.update(result)
            
            if self.option.task_name == "dna690":
                eval_output_dir = self.option.result_dir
                if not os.path.exists(self.option.result_dir): 
                    os.makedirs(self.option.result_dir)
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "a") as writer:

                if self.option.task_name[:3] == "dna" or self.option.task_name == "siamese":
                    eval_result = self.option.data_dir.split('/')[-1] + " "
                else:
                    eval_result = ""

                self.logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))
                    eval_result = eval_result + str(result[key])[:5] + " "
                writer.write(eval_result + "\n")

        if self.option.do_ensemble_pred:
            return results, eval_task, preds, out_label_ids, probs
        else:
            return results