import os

import numpy as np
import torch
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, LlamaConfig, LlamaForCausalLM,
                          Trainer, TrainingArguments)

from dataset import belle_open_source_500k


def load_model():
    path = "./model_config"
    model_config = LlamaConfig.from_pretrained(path)
    model = LlamaForCausalLM(model_config).cuda()
    print(f'memory usage of model: {model.get_memory_footprint() / (1024 * 1024):.2f} MB')
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        print(f"pass unk_token_id {tokenizer.unk_token_id} to pad_token_id")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        loss_mask = inputs.pop('loss_mask', None)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = loss_mask[..., 1:].contiguous() if loss_mask is not None else None
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss_mask = loss_mask.view(-1) if loss_mask is not None else None

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_mask = loss_mask.to(shift_logits.device) if loss_mask is not None else None

        if loss_mask is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits, shift_labels) * loss_mask
            loss = loss.sum() / loss_mask.sum()
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss


class DataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0] * (max_label_length - len(feature["loss_mask"]))
                if isinstance(feature["loss_mask"], list):
                    feature["loss_mask"] = (
                        feature["loss_mask"] + remainder if padding_side == "right" else remainder + feature["loss_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_mask"] = np.concatenate([feature["loss_mask"], remainder]).astype(np.int64)
                else:
                    feature["loss_mask"] = np.concatenate([remainder, feature["loss_mask"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def main():
    training_args = TrainingArguments(
        output_dir=r"./output",
        per_device_train_batch_size=4,
        remove_unused_columns=False,
        report_to='none'
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"world size {world_size} local rank {local_rank}")

    model, tokenizer = load_model()

    data = belle_open_source_500k("./data/Belle_open_source_200.json", tokenizer, 512)

    train_data = data
    val_data = None

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollator(tokenizer,
                                   pad_to_multiple_of=8,
                                   return_tensors="pt",
                                   padding=True),
    )
    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()
