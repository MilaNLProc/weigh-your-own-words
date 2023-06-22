# GPT-2 FULL CODE FOR CAUSAL LANGUAGE MODELLING

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
# data files are already shuffled for gpt2

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
import wandb
from datasets import load_dataset
from src.training import CustomTrainer, GenerationCallback
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

transformers.logging.set_verbosity_info()

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    apply_ear: Optional[bool] = field(default=False)
    ear_reg_strength: Optional[float] = field(default=None)
    ear_strength_dynamic: Optional[bool] = field(default=False)
    ear_disable_special_tokens: Optional[bool] = field(default=False)
    ear_include_hs: Optional[bool] = field(default=False)
    apply_klreg: Optional[bool] = field(default=False)
    klreg_special_share: Optional[float] = field(default=None)
    klreg_hscn_are_special: Optional[bool] = field(default=None)

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )

    id_terms_file: Optional[str] = field(
        default=None, metadata={"help": "The input identity terms file (a text file)."}
    )

    prej_terms_file: Optional[str] = field(
        default=None, metadata={"help": "The input prejudice terms file (a text file)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."
            if self.id_terms_file is not None:
                extension = self.id_terms_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.prej_terms_file is not None:
                extension = self.prej_terms_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # when the model is GPT2 we need to set this
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_type == "gpt2":
        SPECIAL_TOKENS_DICT = {
            "additional_special_tokens": ["<hatespeech>", "<counternarrative>"],
        }
        # Add these special tokens to the vocabulary and resize model's embeddings:
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    if model_args.model_name_or_path:
        # prev DialoGPT training code uses AutoModelWithLMHead but it's deprecated now
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise NotImplementedError("Should not come here!")
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    # resize is not working with init unfortunately, so we deepcopy an initialized model instead.
    # We need to find a way to successfully update model config in this class
    import copy

    def model_init():
        return copy.deepcopy(model)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def find_sub_list(sl: List, l: List):
        """Find the spans a sublist is found in a list.
        Parameters
        ----------
        sl : List
            Sublist to search.
        l : List
            List to search "sl" in.
        Returns
        -------
        List[Tuple[int,int]]
            List of tuples, each containing start and end index of the span found.
        """

        results = list()
        sll = len(sl)

        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind : ind + sll] == sl:
                results.append((ind, ind + sll))

        return results

    list_id_terms = None
    if data_args.id_terms_file is not None:
        # we add leading space so that we can treat all words as non initial words (GPT tokenizers has the distinction, see https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer.add_prefix_space)
        with open(data_args.id_terms_file) as fp:
            list_id_terms = [f" {f.strip()}" for f in fp.readlines()]
            list_id_terms = [tokenizer.encode(t) for t in list_id_terms]

    list_prej_terms = None
    if data_args.prej_terms_file is not None:
        with open(data_args.prej_terms_file) as fp:
            list_prej_terms = [f" {f.strip()}" for f in fp.readlines()]
            list_prej_terms = [tokenizer.encode(t) for t in list_prej_terms]

    def tokenize_dialogpt(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ]

        # separate context from response and encode them separately
        texts = list()
        for ex in examples[text_column_name]:
            clean_ex = ex.strip("<hatespeech> ").strip(" <|endoftext|>")
            context, response = [t.strip() for t in clean_ex.split("<counternarrative>")]
            texts.append(
                f"{context}{tokenizer.eos_token}{response}{tokenizer.eos_token}"
            )

        tokens = tokenizer(texts, padding="max_length", max_length=data_args.block_size)
        tokens["labels"] = tokens["input_ids"].copy()

        # In the original training script inputs longer than block size are discarded
        # The previous tokenization doesn't trucate inputs, then we can have inputs larger than block_size
        tokens = {
            k: [toks for toks in v if len(toks) <= data_args.block_size]
            for k, v in tokens.items()
        }

        # We add additional info the batch with the keys "type_ids" and "term_type_ids"
        tokens["type_ids"] = list()

        for t in examples[text_column_name]:
            clean_ex = ex.strip("<hatespeech> ").strip(" <|endoftext|>")
            context, response = [t.strip() for t in clean_ex.split("<counternarrative>")]

            hs_token_count = len(tokenizer.encode(f"{context}{tokenizer.eos_token}"))
            cn_token_count = len(tokenizer.encode(f"{response}{tokenizer.eos_token}"))

            tokens["type_ids"].append(
                [0] * hs_token_count
                + [1] * cn_token_count
                + [2] * (data_args.block_size - (hs_token_count + cn_token_count))
            )

        if list_id_terms or list_prej_terms:
            raise NotImplementedError(
                "Per term type EAR loss not implemented yet for dialogpt"
            )
            # tokens["term_type_ids"] = list()

        return tokens

    # Here how we handled the line by line basically padding: https://finisky.github.io/finetunelmlinebyline.en/456G
    def tokenize_gpt2(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ]
        tokens = tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=data_args.block_size,
            return_special_tokens_mask=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()

        # We add additional info the batch with the keys "type_ids" and "term_type_ids"
        tokens["type_ids"] = list()

        if list_id_terms or list_prej_terms:
            tokens["term_type_ids"] = list()

        for t in examples[text_column_name]:

            # include type ids, where 0: source hs, 1: counternarrative tokens, 2: pad
            hs_text = t.split("<counternarrative>")[0] + "<counternarrative>"
            cn_text = t.split("<counternarrative>")[1]
            hs_token_count = len(tokenizer.encode(hs_text))
            cn_token_count = len(tokenizer.encode(cn_text))
            tokens["type_ids"].append(
                [0] * hs_token_count
                + [1] * cn_token_count
                + [2] * (data_args.block_size - (hs_token_count + cn_token_count))
            )

            if "term_type_ids" in tokens:
                # include term type, where 0: normal, 1: identity term, 2: prejudice term
                token_ids = tokenizer.encode(
                    f" {t.lower()}"
                )  # remember the leading space trick

                term_types = [0] * len(token_ids)

                if list_id_terms:
                    identity_terms_spans = [
                        find_sub_list(id_tokens, token_ids)
                        for id_tokens in list_id_terms
                    ]
                    for id_spans in identity_terms_spans:
                        for id_span in id_spans:
                            for i in range(id_span[0], id_span[1]):
                                term_types[i] = 1

                if list_prej_terms:
                    prejudice_terms_spans = [
                        find_sub_list(id_tokens, token_ids)
                        for id_tokens in list_prej_terms
                    ]
                    for id_spans in prejudice_terms_spans:
                        for id_span in id_spans:
                            for i in range(id_span[0], id_span[1]):
                                term_types[i] = 2

                tokens["term_type_ids"].append(
                    term_types + [0] * (data_args.block_size - len(term_types))
                )

        return tokens

    tfn = tokenize_gpt2 if model_args.model_type == "gpt2" else tokenize_dialogpt
    tokenized_datasets = datasets.map(
        tfn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Hyperparameter search
    # -----
    def cn_hp_space(trial):

        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
            ),
            # trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [2, 4]
            ),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5, log=True),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0, 0.1])
            # "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
            # "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
        }

    # ----
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    callbacks = [EarlyStoppingCallback(3, 1e-4)]
    if model_args.model_type == "gpt2":  # we have it only for gpt2 atm
        callbacks.append(GenerationCallback())

    training_args.remove_unused_columns = False

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        model_type=model_args.model_type,
        # model_init=model_init,
        args=training_args,
        train_dataset=lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        apply_ear=model_args.apply_ear,
        ear_reg_strength=model_args.ear_reg_strength,
        ear_disable_special_tokens=model_args.ear_disable_special_tokens,
        ear_strength_dynamic=model_args.ear_strength_dynamic,
        ear_include_hs=model_args.ear_include_hs,
        callbacks=callbacks,
        apply_klreg=model_args.apply_klreg,
        klreg_special_share=model_args.klreg_special_share,
        klreg_hscn_are_special=model_args.klreg_hscn_are_special,
    )
    # trainer.add_callback(
    #    EarlyStoppingCallback(1, 0.0001))  # we can change the criteria, in transformers test file it is 0.0001

    # best_run = trainer.hyperparameter_search(n_trials=10, direction="minimize", hp_space=cn_hp_space)

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(
            model_args.model_name_or_path
        ):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_clm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
