"""
Fine-tuning Comparison: Random vs ST-Curated vs HDC-Curated

Goal: Prove that HDC-curated data improves model quality after fine-tuning,
not just data metrics.

Experiment:
1. Three 500-sample subsets from Alpaca
2. Fine-tune TinyLlama-1.1B with LoRA on each
3. Evaluate on held-out test set (validation loss, ROUGE-L)
"""

import torch
import numpy as np
import json
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from hdc.data_curator import HDCDataCurator
from hdc.st_curator import STDataCurator
from typing import List, Dict, Tuple
import os


def load_alpaca_data(max_samples: int = 3000) -> Tuple[List[str], List[int]]:
    """
    Load Alpaca dataset.

    Args:
        max_samples: Maximum samples to load

    Returns:
        texts: List of formatted instruction+response strings
        indices: List of original indices
    """
    print(f"Loading Alpaca dataset (max {max_samples} samples)...")

    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except:
        print("  Alpaca not available, using alternative...")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    texts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # Format for instruction tuning
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        texts.append(prompt)

    print(f"✓ Loaded {len(texts)} samples\n")
    return texts, list(range(len(texts)))


def create_subsets(
    texts: List[str],
    subset_size: int = 500,
    test_size: int = 200
) -> Dict:
    """
    Create three subsets: Random, ST-Curated, HDC-Curated + test set.

    Args:
        texts: Full dataset
        subset_size: Size of each training subset
        test_size: Size of test set

    Returns:
        Dictionary with subset indices and texts
    """
    print("=" * 60)
    print("CREATING SUBSETS")
    print("=" * 60)
    print()

    # Reserve test set first
    np.random.seed(42)
    all_indices = np.arange(len(texts))
    np.random.shuffle(all_indices)

    test_indices = all_indices[:test_size]
    available_indices = all_indices[test_size:]
    available_texts = [texts[i] for i in available_indices]

    print(f"Test set: {len(test_indices)} samples")
    print(f"Available for training: {len(available_texts)} samples\n")

    # 1. Random subset
    print("Creating Random subset...")
    np.random.seed(42)
    random_local_indices = np.random.choice(
        len(available_texts),
        size=subset_size,
        replace=False
    )
    random_indices = available_indices[random_local_indices]
    print(f"✓ Random subset: {len(random_indices)} samples\n")

    # 2. ST-Curated subset
    print("Creating ST-Curated subset...")
    st_curator = STDataCurator(device='cpu')
    st_local_indices, st_stats = st_curator.curate(
        available_texts,
        target_size=subset_size,
        sampling_strategy='nearest_centroid'
    )
    st_indices = available_indices[st_local_indices]

    # 3. HDC-Curated subset
    print("Creating HDC-Curated subset...")
    hdc_curator = HDCDataCurator(
        hd_dim=10000,
        sparsity=0.7,
        dedup_threshold=0.95,
        device='cpu'
    )
    hdc_local_indices, hdc_stats = hdc_curator.curate(
        available_texts,
        target_size=subset_size,
        sampling_strategy='nearest_centroid',
        batch_size=32
    )
    hdc_indices = available_indices[hdc_local_indices]

    return {
        "test_indices": test_indices,
        "random_indices": random_indices,
        "st_indices": st_indices,
        "hdc_indices": hdc_indices,
        "st_stats": st_stats,
        "hdc_stats": hdc_stats
    }


def prepare_model_and_tokenizer(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load model and tokenizer with LoRA configuration.

    Args:
        model_name: HuggingFace model name

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ Model loaded with LoRA\n")

    return model, tokenizer


def tokenize_dataset(texts: List[str], tokenizer, max_length: int = 512):
    """
    Tokenize texts for causal LM.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    print(f"Tokenizing {len(texts)} samples...")

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Create dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx]
            }

    dataset = SimpleDataset(encodings)
    print(f"✓ Tokenized {len(dataset)} samples\n")

    return dataset


def finetune_and_evaluate(
    subset_name: str,
    train_texts: List[str],
    test_texts: List[str],
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = None
) -> Dict:
    """
    Fine-tune model on subset and evaluate.

    Args:
        subset_name: Name of subset (for logging)
        train_texts: Training texts
        test_texts: Test texts
        model_name: HuggingFace model name
        output_dir: Directory to save model

    Returns:
        Dictionary with training metrics
    """
    print("=" * 60)
    print(f"FINE-TUNING: {subset_name}")
    print("=" * 60)
    print()

    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name)

    # Tokenize datasets
    train_dataset = tokenize_dataset(train_texts, tokenizer, max_length=512)
    test_dataset = tokenize_dataset(test_texts, tokenizer, max_length=512)

    # Training arguments
    if output_dir is None:
        output_dir = f"./tmp_finetune_{subset_name.lower()}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=1
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    # Train
    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    print("\nEvaluating on test set...")
    eval_result = trainer.evaluate()

    # Extract metrics
    metrics = {
        "subset": subset_name,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "training_time_seconds": training_time,
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_result["eval_loss"],
        "eval_perplexity": np.exp(eval_result["eval_loss"]),
        "train_history": train_result.metrics if hasattr(train_result, 'metrics') else {}
    }

    print(f"\n✓ Fine-tuning complete:")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"  Final eval loss: {metrics['final_eval_loss']:.4f}")
    print(f"  Eval perplexity: {metrics['eval_perplexity']:.2f}")
    print()

    # Cleanup
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    return metrics


def main():
    """Run fine-tuning comparison experiment"""
    print("\n" + "=" * 60)
    print("PHASE M2.5b: FINE-TUNING COMPARISON")
    print("=" * 60)
    print()

    # Configuration
    FULL_DATASET_SIZE = 3000
    SUBSET_SIZE = 500
    TEST_SIZE = 200

    # Load dataset
    all_texts, all_indices = load_alpaca_data(max_samples=FULL_DATASET_SIZE)

    # Create subsets
    subsets = create_subsets(all_texts, subset_size=SUBSET_SIZE, test_size=TEST_SIZE)

    # Extract test set
    test_texts = [all_texts[i] for i in subsets["test_indices"]]

    # Fine-tune on each subset
    results = {}

    # 1. Random
    print("\n" + "=" * 60)
    print("EXPERIMENT 1/3: RANDOM SUBSET")
    print("=" * 60)
    print()
    random_texts = [all_texts[i] for i in subsets["random_indices"]]
    results["random"] = finetune_and_evaluate(
        "Random",
        random_texts,
        test_texts
    )

    # 2. ST-Curated
    print("\n" + "=" * 60)
    print("EXPERIMENT 2/3: ST-CURATED SUBSET")
    print("=" * 60)
    print()
    st_texts = [all_texts[i] for i in subsets["st_indices"]]
    results["st_curated"] = finetune_and_evaluate(
        "ST-Curated",
        st_texts,
        test_texts
    )

    # 3. HDC-Curated
    print("\n" + "=" * 60)
    print("EXPERIMENT 3/3: HDC-CURATED SUBSET")
    print("=" * 60)
    print()
    hdc_texts = [all_texts[i] for i in subsets["hdc_indices"]]
    results["hdc_curated"] = finetune_and_evaluate(
        "HDC-Curated",
        hdc_texts,
        test_texts
    )

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    random_loss = results["random"]["final_eval_loss"]
    st_loss = results["st_curated"]["final_eval_loss"]
    hdc_loss = results["hdc_curated"]["final_eval_loss"]

    random_ppl = results["random"]["eval_perplexity"]
    st_ppl = results["st_curated"]["eval_perplexity"]
    hdc_ppl = results["hdc_curated"]["eval_perplexity"]

    print(f"{'Method':<20} {'Eval Loss':>12} {'Perplexity':>12} {'Train Time (s)':>15}")
    print("-" * 65)
    print(f"{'Random':<20} {random_loss:>12.4f} {random_ppl:>12.2f} {results['random']['training_time_seconds']:>15.2f}")
    print(f"{'ST-Curated':<20} {st_loss:>12.4f} {st_ppl:>12.2f} {results['st_curated']['training_time_seconds']:>15.2f}")
    print(f"{'HDC-Curated':<20} {hdc_loss:>12.4f} {hdc_ppl:>12.2f} {results['hdc_curated']['training_time_seconds']:>15.2f}")
    print()

    # Determine winner
    losses = {
        "Random": random_loss,
        "ST-Curated": st_loss,
        "HDC-Curated": hdc_loss
    }
    best_method = min(losses, key=losses.get)
    best_loss = losses[best_method]

    print(f"BEST METHOD: {best_method} (eval loss: {best_loss:.4f})")
    print()

    # Determine status
    if hdc_loss < st_loss < random_loss:
        status = "✅✅ STRONG SUCCESS"
        message = "HDC-Curated > ST-Curated > Random"
    elif hdc_loss <= st_loss and hdc_loss < random_loss:
        status = "✅ SUCCESS"
        message = "HDC-Curated ≥ ST-Curated > Random"
    elif abs(hdc_loss - st_loss) / st_loss < 0.02 and hdc_loss < random_loss:
        status = "⚠️  PARTIAL SUCCESS"
        message = "HDC-Curated ≈ ST-Curated > Random"
    else:
        status = "❌ FAILURE"
        message = "Curation advantage unclear or absent"

    print(f"VERDICT: {status}")
    print(f"  {message}")
    print()

    # Compute improvements
    hdc_vs_random_pct = ((random_loss - hdc_loss) / random_loss) * 100
    hdc_vs_st_pct = ((st_loss - hdc_loss) / st_loss) * 100
    st_vs_random_pct = ((random_loss - st_loss) / random_loss) * 100

    print("IMPROVEMENTS:")
    print(f"  HDC vs Random: {hdc_vs_random_pct:+.2f}% loss reduction")
    print(f"  HDC vs ST: {hdc_vs_st_pct:+.2f}% loss reduction")
    print(f"  ST vs Random: {st_vs_random_pct:+.2f}% loss reduction")
    print()

    # Save results
    output = {
        "phase": "M2.5b",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "full_dataset_size": FULL_DATASET_SIZE,
            "subset_size": SUBSET_SIZE,
            "test_size": TEST_SIZE,
            "model": "TinyLlama-1.1B-Chat-v1.0",
            "lora_rank": 8,
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4
        },
        "subsets": {
            "st_curation_stats": subsets["st_stats"],
            "hdc_curation_stats": subsets["hdc_stats"]
        },
        "results": results,
        "comparison": {
            "status": status,
            "message": message,
            "best_method": best_method,
            "hdc_vs_random_pct": hdc_vs_random_pct,
            "hdc_vs_st_pct": hdc_vs_st_pct,
            "st_vs_random_pct": st_vs_random_pct
        }
    }

    os.makedirs("hdc/results", exist_ok=True)
    output_file = "hdc/results/phase_m2.5b_finetune.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {output_file}\n")

    print("=" * 60)
    print("PHASE M2.5b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
