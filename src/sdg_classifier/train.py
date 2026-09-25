"""Fine-tune a transformer for multi-label SDG classification, fully or with LoRA."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from .data import SDG_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "distilbert/distilbert-base-uncased"


class WeightedTrainer(Trainer):
    """Trainer with a BCE loss that weighs the positive examples of rare labels up.

    Setting ``model.loss_fct`` or ``model.config.loss_fn``, as the 2024 scripts did, has no effect: the Hugging Face
    models build their own unweighted loss. The weighted loss has to be computed here.
    """

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.pos_weight is None else self.pos_weight.to(outputs.logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels, pos_weight=weight)
        return (loss, outputs) if return_outputs else loss


def multi_label_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, probs, average="micro")
    except ValueError:  # a label without positive examples, e.g. in a tiny test run
        roc_auc = float("nan")
    return {
        "f1": f1,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(labels, preds),  # exact match of all 17 labels
    }


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(logits, p.label_ids)


def encode(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize(batch):
        encoding = tokenizer(batch["abstract"], truncation=True, max_length=max_length)
        encoding["labels"] = np.stack([batch[c] for c in SDG_COLUMNS], axis=1).astype(np.float32).tolist()
        return encoding

    return dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)


def train(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    lora: bool = False,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float | None = None,
    max_length: int = 128,
    class_weights: bool = True,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Train, keep the best epoch by micro F1, save the model to ``output_dir`` and return the test metrics."""
    dataset = DatasetDict.load_from_disk(str(dataset_dir))
    if limit:
        dataset = DatasetDict({k: v.select(range(min(limit, len(v)))) for k, v in dataset.items()})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoded = encode(dataset, tokenizer, max_length).with_format("torch")

    id2label = dict(enumerate(SDG_COLUMNS))
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        problem_type="multi_label_classification",
        num_labels=len(SDG_COLUMNS),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    if lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            task_type="SEQ_CLS",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_lin", "v_lin"],  # DistilBERT's query and value projections
            modules_to_save=["pre_classifier", "classifier"],  # the new, randomly initialised head is trained fully
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    pos_weight = None
    if class_weights:
        positives = torch.tensor(dataset["train"].to_pandas()[SDG_COLUMNS].sum().to_numpy(), dtype=torch.float32)
        pos_weight = (len(dataset["train"]) - positives) / positives.clamp(min=1)
        log.info(
            "Positive class weights: %s",
            {c: round(w, 1) for c, w in zip(SDG_COLUMNS, pos_weight.tolist(), strict=True)},
        )

    checkpoints = output_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate or (2e-4 if lora else 2e-5),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=0.06,  # a float is a ratio of all steps
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )
    trainer.train()

    validation = trainer.evaluate()
    test = trainer.evaluate(encoded["test"], metric_key_prefix="test")

    final = trainer.model.merge_and_unload() if lora else trainer.model
    final.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    shutil.rmtree(checkpoints, ignore_errors=True)
    (output_dir / "metrics.json").write_text(json.dumps({**validation, **test}, indent=2) + "\n")
    return test
