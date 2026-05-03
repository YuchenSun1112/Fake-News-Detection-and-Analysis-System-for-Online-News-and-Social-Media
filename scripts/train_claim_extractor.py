import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    cleaned = []
    for record in records:
        input_text = str(record.get("input_text", "")).strip()
        target_text = str(record.get("target_text", "")).strip()
        if input_text and target_text:
            cleaned.append({"input_text": input_text, "target_text": target_text})

    if not cleaned:
        raise ValueError(f"No usable training records found in {path}")

    return cleaned


def tokenize_dataset(dataset, tokenizer, max_source_length: int, max_target_length: int):
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a seq2seq claim extractor on input_text -> target_text JSON data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "claim_extraction_dataset.json",
    )
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "claim_extractor" / "final",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision. Disabled by default because small seq2seq runs can produce NaN gradients.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision on supported GPUs.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--debug-sample-size",
        type=int,
        default=None,
        help="Use only the first N records for a quick smoke test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.dataset)
    if args.debug_sample_size:
        records = records[: args.debug_sample_size]

    raw_dataset = Dataset.from_list(records)
    split_dataset = raw_dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized_dataset = tokenize_dataset(
        split_dataset,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    train_label_lengths = [
        sum(token_id != tokenizer.pad_token_id for token_id in labels)
        for labels in tokenized_dataset["train"]["labels"][:20]
    ]
    if not train_label_lengths or min(train_label_lengths) <= 0:
        raise ValueError("Tokenized labels are empty; check target_text in the dataset.")
    print(
        "Tokenized label length check:",
        {
            "sample_size": len(train_label_lengths),
            "min": min(train_label_lengths),
            "max": max(train_label_lengths),
        },
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["test"],
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation:", metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Claim extractor model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
