import os
from functools import lru_cache
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertConfig,
    EarlyStoppingCallback,
)

from config import BASELINE_MODEL_NAME, BASELINE_MODEL_DIR, MAX_LENGTH
from src.data_loader import load_baseline_data


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_baseline(debug_sample_size=None):
    df = load_baseline_data()

    if debug_sample_size is not None:
        df = df.head(debug_sample_size)

    train_df, test_df = train_test_split(
        df[["final_text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df = train_df.rename(columns={"final_text": "text"})
    test_df = test_df.rename(columns={"final_text": "text"})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = BertTokenizer.from_pretrained(BASELINE_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    config = BertConfig.from_pretrained(
        BASELINE_MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
    )

    model = BertForSequenceClassification.from_pretrained(
        BASELINE_MODEL_NAME,
        config=config,
    )

    training_args = TrainingArguments(
        output_dir=BASELINE_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=os.path.join(BASELINE_MODEL_DIR, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Training baseline classifier...")
    trainer.train()

    results = trainer.evaluate()
    print("Final evaluation:", results)

    final_dir = os.path.join(BASELINE_MODEL_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Baseline model saved to {final_dir}")


@lru_cache(maxsize=1)
def load_baseline_model():
    model_path = os.path.join(BASELINE_MODEL_DIR, "final")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_baseline(text: str):
    tokenizer, model, device = load_baseline_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    prob_fake = probs[0].item()
    prob_real = probs[1].item()

    label = "real" if prob_real >= prob_fake else "fake"
    confidence = max(prob_real, prob_fake)

    return {
        "label": label,
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
    }
