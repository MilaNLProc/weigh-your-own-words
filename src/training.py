import os
import pdb
from logging import getLogger

import torch
import wandb
from einops import rearrange, reduce, repeat
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import Trainer, TrainerCallback, TrainingArguments

logger = getLogger(__name__)


def compute_kldiv(inputs, attention_mask, term_type_ids, klreg_special_share):
    # assert inputs.ndim == 4, "Here we expect 4 dimensions in the form L,B,H,S"
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form L,B,H,S,S"

    L, B, H, S, S = inputs.shape
    # import pdb; pdb.set_trace()
    B, S = attention_mask.shape
    device = inputs.device

    # create a causal (autoregressive) mask
    causal_mask = torch.tril(
        torch.ones((S, S), device=device, dtype=attention_mask.dtype)
    )

    # repeat padding mask for the entire sequence
    padding_mask = repeat(attention_mask, "b s -> b c s", c=S)

    # mask is padding AND causal masks
    mask = padding_mask & causal_mask  # shape: B, S, S

    inputs = rearrange(inputs, "l b h s1 s2 -> l h b s1 s2")

    #  first, average over attention heads. TODO we could try to average this lateer strategies
    # average inputs across heads
    inputs = reduce(inputs, "l h b s1 s2 -> l b s1 s2", "mean")

    # masked softmax is needed because attention weights do not sum to one after averaging across heads
    weights = inputs.masked_fill(~mask.bool(), torch.finfo(inputs.dtype).min)
    weights = weights.softmax(-1)

    is_special_mask = term_type_ids > 0
    prev_special_count = is_special_mask.cumsum(-1)
    special_total_share = klreg_special_share

    # TODO looping across sequence here, can't think of a better way to do it
    target_dists = list()
    for l in range(L):
        for i in range(B):
            for j in range(S):
                token_count = j + 1
                special_token_count = prev_special_count[i, j].item()
                normal_token_count = token_count - special_token_count

                # total attention to give to special tokens set and normal token set
                special_total = special_total_share if special_token_count != 0 else 0
                normal_total = 1 - special_total

                # this happens if we are on the first step and we are considering HS and CN tokens as specials
                if normal_token_count == 0:
                    target_dist = torch.full((S,), 1.0, device=device)

                # in all other cases, we can apply the standard share splitting to special and normal tokens
                else:
                    # target attention value for a normal token
                    normal_target = normal_total / normal_token_count
                    target_dist = torch.full((S,), normal_target, device=device)

                    if special_token_count > 0:
                        # target attention value for a special token
                        special_target = special_total / special_token_count
                        target_dist = target_dist.masked_fill(
                            is_special_mask[i], special_target
                        )

                target_dists.append(target_dist)

    target_dists = torch.stack(target_dists).view(L, B, S, S)

    # mask target distribution with causal mask
    target_dists *= mask

    # compute kl divergence for each step. We sum pointwise contribution manually: there is hence no need to take care of padding steps since will have distance = 0 to attention weights
    weights = weights.masked_fill(weights == 0.0, 1.0)
    target_dists = target_dists.masked_fill(target_dists == 0.0, 1.0)

    kl_loss = torch.nn.functional.kl_div(weights.log(), target_dists, reduction="none")

    kl_loss = reduce(kl_loss, "l b s1 s2 -> l b s1", "sum")

    # to stay coherent with entropy loss, take the avg across layers
    kl_loss = reduce(kl_loss, "l b s -> b s", "mean")

    return kl_loss


def compute_negative_entropy(
    inputs, attention_mask: torch.torch, return_values: bool = False
):
    """Compute the negative entropy across layers of a network for given inputs.
    Adapted from: https://github.com/g8a9/ear/blob/master/ear/__init__.py

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: B,H, S
        - attention_mask. Tensor with dim: B, S
    """
    # assert inputs.ndim == 4, "Here we expect 4 dimensions in the form L,B,H,S"
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form L,B,H,S,S"

    # import pdb; pdb.set_trace()
    B, S = attention_mask.shape
    device = inputs.device

    # create a causal (autoregressive) mask
    causal_mask = torch.tril(
        torch.ones((S, S), device=device, dtype=attention_mask.dtype)
    )

    # repeat padding mask for the entire sequence
    padding_mask = repeat(attention_mask, "b s -> b c s", c=S)

    # mask is padding AND causal masks
    mask = padding_mask & causal_mask  # shape: B, S, S

    inputs = rearrange(inputs, "l b h s1 s2 -> l h b s1 s2")

    #  first, average over attention heads. TODO we could try to average this lateer strategies
    # average inputs across heads
    inputs = reduce(inputs, "l h b s1 s2 -> l b s1 s2", "mean")

    # masked softmax is needed because attention weights do not sum to one after averaging across heads
    weights = inputs.masked_fill(~mask.bool(), torch.finfo(inputs.dtype).min)
    weights = weights.softmax(-1)

    # neg_entr = reduce(weights * torch.log(weights), "l b s1 s2 -> l b s1", "sum")
    cat = torch.distributions.Categorical(probs=weights)

    # compute entropies for each layer and sequence step separately
    neg_entr = -cat.entropy()  # l b s1

    # avg across layers
    neg_entr = reduce(neg_entr, "l b s -> b s", "mean")

    return neg_entr


class GenerationCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        if os.path.exists("./data/monitored_samples.txt"):
            with open("./data/monitored_samples.txt") as fp:
                self.monitored_samples = [f.strip() for f in fp.readlines()]
        else:
            self.monitored_samples = list()

        self.decoder_max_len = 156

    def on_evaluate(self, args: TrainingArguments, state, control, **kwargs):
        logs = dict()
        m = kwargs["model"]
        t = kwargs["tokenizer"]

        if self.monitored_samples:
            logger.info(
                f"Generating CNs for {len(self.monitored_samples)} monitored samples."
            )
            table_rows = list()

            for sample in tqdm(self.monitored_samples, desc="Monitored Sample"):

                hs = "<hatespeech> " + sample + " <counternarrative>"
                encoded_hs_ids = t(
                    hs, truncation=True, padding=True, return_tensors="pt"
                ).to(m.device)

                try:
                    bs_generation = m.generate(
                        **encoded_hs_ids,
                        max_length=self.decoder_max_len,
                        num_beams=5,
                        early_stopping=True,
                        num_return_sequences=1,
                        repetition_penalty=2.0,
                        do_sample=False,
                    )
                    gen_text = t.batch_decode(bs_generation, skip_special_tokens=False)[
                        0
                    ]
                    gen_text = " ".join(gen_text.split("<counternarrative>")[1:])
                except:
                    gen_text = "error in generation"

                table_rows.append([sample, gen_text, "BS", state.global_step])

                try:
                    tk_generation = m.generate(
                        **encoded_hs_ids,
                        max_length=self.decoder_max_len,
                        do_sample=True,
                        top_k=40,
                        num_return_sequences=1,
                    )
                    gen_text = t.batch_decode(tk_generation, skip_special_tokens=False)[
                        0
                    ]
                    gen_text = " ".join(gen_text.split("<counternarrative>")[1:])
                except:
                    gen_text = "error in generation"

                table_rows.append([sample, gen_text, "TK", state.global_step])

        logs.update(
            {
                "monitored_samples": wandb.Table(
                    columns=["query", "response", "decoding", "step"],
                    rows=table_rows,
                )
            }
        )

        wandb.log(logs)


class CustomTrainer(Trainer):
    """Custom Trainer for a custom loss function."""

    def __init__(self, *args, **kwargs):
        apply_ear = kwargs.pop("apply_ear", False)
        ear_reg_strength = kwargs.pop("ear_reg_strength", None)

        self.model_type = kwargs.pop("model_type")
        self.ear_disable_special_tokens = kwargs.pop("ear_disable_special_tokens", False)
        self.ear_strength_dynamic = kwargs.pop("ear_strength_dynamic", False)
        self.apply_ear = apply_ear
        self.ear_include_hs = kwargs.pop("ear_include_hs", False)
        self.apply_klreg = kwargs.pop("apply_klreg", False)
        self.klreg_special_share = kwargs.pop("klreg_special_share")
        self.klreg_hscn_are_special = kwargs.pop("klreg_hscn_are_special")

        if apply_ear and ear_reg_strength is None:
            raise ValueError("Specify a regularization strength")

        self.ear_reg_strength = ear_reg_strength

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Add the EAR penalization term to the standard loss.

        Note:
        - we regularize only generation steps (i.e., type_ids == 1)
        - if the batch contains "term_type_ids", we regularize only those steps in the generation sequence (i.e., term_type_ids == 1 for identity terms, == 2 for prejudice terms, or both).
        """

        type_ids = inputs.pop("type_ids", None)
        term_type_ids = inputs.pop("term_type_ids", None)
        # special_tokens_mask = inputs.pop("special_tokens_mask")

        logging_metrics = dict()

        outputs = model(**inputs, output_attentions=True)
        loss = outputs["loss"]

        is_training = loss.requires_grad

        if is_training and self.apply_klreg:
            attn = torch.stack(outputs["attentions"])
            attn_mask = inputs["attention_mask"]

            # if we want HS an CN special tokens to receive part of the special tokens attention share, we set them accordingly
            if self.klreg_hscn_are_special:

                hs_token = self.tokenizer.vocab["<hatespeech>"]
                cn_token = self.tokenizer.vocab["<counternarrative>"]

                term_type_ids = term_type_ids.masked_fill(
                    torch.logical_or(
                        inputs["input_ids"] == hs_token,
                        inputs["input_ids"] == cn_token,
                    ),
                    1,  # any greater than zero value will do the job
                )

            kl_loss = compute_kldiv(
                attn, attn_mask, term_type_ids, self.klreg_special_share
            )

            # we consider only generation steps for the regularization term
            mask = (type_ids == 1).bool()

            if self.ear_include_hs:
                mask = torch.logical_or(mask, (type_ids == 0).bool())

            kl_loss = kl_loss[mask].mean()

            strength_scheduler = (
                (1 - self.state.global_step / self.state.max_steps)
                if self.ear_strength_dynamic
                else 1.0
            )
            final_kl_loss = strength_scheduler * self.ear_reg_strength * kl_loss

            loss += final_kl_loss

            if (
                self.state.global_step % self.args.logging_steps == 0
            ) and loss.requires_grad:
                logging_metrics["kl/loss"] = kl_loss.item()
                logging_metrics["kl/scaled_loss"] = final_kl_loss.item()
                logging_metrics["kl/reg_strength"] = (
                    strength_scheduler * self.ear_reg_strength
                )
                wandb.log(
                    {"train/token_count": wandb.Histogram(mask.sum(-1).tolist())},
                    step=self.state.global_step,
                )
                self.log(logging_metrics)

        if is_training and self.apply_ear:
            """
            We will compute one EAR loss component for every time step.
            Note that we are considering every step equally.
            """
            attn = torch.stack(outputs["attentions"])
            attn_mask = inputs["attention_mask"]

            if self.ear_disable_special_tokens:

                if self.model_type == "dialogpt":
                    raise NotImplementedError("Can't use this parameter with dialogpt")

                hs_token = self.tokenizer.vocab["<hatespeech>"]
                cn_token = self.tokenizer.vocab["<counternarrative>"]
                tokens_to_mask = [hs_token, cn_token]

                # import pdb; pdb.set_trace()
                for t in tokens_to_mask:
                    attn_mask = torch.where(inputs["input_ids"] != t, attn_mask, 0)

            strength_scheduler = (
                (1 - self.state.global_step / self.state.max_steps)
                if self.ear_strength_dynamic
                else 1.0
            )
            negative_entropy = compute_negative_entropy(attn, attn_mask)

            # we consider only generation steps for the regularization term
            mask = (type_ids == 1).bool()

            if self.ear_include_hs:
                mask = torch.logical_or(mask, (type_ids == 0).bool())

            if term_type_ids is not None:
                # only identity or prejudice terms contribute to the reg term
                mask = torch.logical_and(
                    mask,
                    torch.logical_or(
                        (term_type_ids == 1).bool(), (term_type_ids == 2).bool()
                    ),
                )

            negative_entropy = negative_entropy[mask].mean()

            # import pdb; pdb.set_trace()
            ear_loss = strength_scheduler * self.ear_reg_strength * negative_entropy

            loss += ear_loss

            if (
                self.state.global_step % self.args.logging_steps == 0
            ) and loss.requires_grad:
                logging_metrics["ear/entropy"] = -negative_entropy.item()
                logging_metrics["ear/loss"] = ear_loss.item()
                logging_metrics["ear/reg_strength"] = (
                    strength_scheduler * self.ear_reg_strength
                )
                wandb.log(
                    {"train/ear/token_count": wandb.Histogram(mask.sum(-1).tolist())},
                    step=self.state.global_step,
                )
                self.log(logging_metrics)

        return (loss, outputs) if return_outputs else loss
