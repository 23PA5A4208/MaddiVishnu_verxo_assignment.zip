"""
GSM8K Training Script with LLaMA 3.2 1B and LoRA
Vexoo Labs AI Engineer Assignment - Part 2

This script implements:
- Data loading and train/test split from GSM8K dataset
- Tokenization compatible with LLaMA architecture
- LoRA-based Supervised Fine-Tuning (SFT)
- Training loop with logging
- Evaluation using exact match accuracy
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from tqdm import tqdm
import re

# Configuration
class Config:
    """Training configuration"""
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # LLaMA 3.2 1B
    DATASET_NAME = "openai/gsm8k"

    # Training parameters
    TRAIN_SAMPLES = 3000
    EVAL_SAMPLES = 1000
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4

    # LoRA parameters
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # Paths
    OUTPUT_DIR = "./gsm8k_lora_output"
    LOG_FILE = "./training_logs.json"

class GSM8KDataProcessor:
    """Handles GSM8K dataset loading and preprocessing"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_and_split(self):
        """Load GSM8K and create train/test split"""
        print("📚 Loading GSM8K dataset from Hugging Face...")

        # Load full dataset
        dataset = load_dataset(Config.DATASET_NAME, "main")

        # GSM8K has 'train' (7473 samples) and 'test' (1319 samples)
        # We'll use 3000 from train for training, 1000 from test for evaluation
        train_data = dataset['train'].select(range(min(Config.TRAIN_SAMPLES, len(dataset['train']))))
        eval_data = dataset['test'].select(range(min(Config.EVAL_SAMPLES, len(dataset['test']))))

        print(f"✅ Loaded {len(train_data)} training samples")
        print(f"✅ Loaded {len(eval_data)} evaluation samples")

        return train_data, eval_data

    def format_example(self, example):
        """Format GSM8K example for LLaMA training"""
        # GSM8K format: question -> chain-of-thought -> final answer
        question = example['question']
        answer = example['answer']

        # Extract final answer (after ####)
        final_answer = answer.split("####")[-1].strip()

        # Create instruction-following format
        formatted = f"""### Instruction:
Solve the following math problem step by step.

### Problem:
{question}

### Solution:
{answer}

### Final Answer:
{final_answer}
"""
        return formatted

    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        # Format all examples
        texts = [self.format_example({"question": q, "answer": a}) 
                for q, a in zip(examples['question'], examples['answer'])]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None
        )

        # Labels same as input_ids for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

class LoRAModelSetup:
    """Sets up LLaMA 3.2 1B with LoRA configuration"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Load model and apply LoRA"""
        print(f"🚀 Loading {Config.MODEL_NAME}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in 4-bit for memory efficiency (QLoRA-style)
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare model for k-bit training
        from peft import prepare_model_for_kbit_training
        self.model = prepare_model_for_kbit_training(self.model)

        print("🔧 Applying LoRA configuration...")

        # LoRA config targeting attention layers
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

        return self.model, self.tokenizer

class GSM8KTrainer:
    """Training loop with logging and evaluation"""

    def __init__(self, model, tokenizer, train_data, eval_data):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.eval_data = eval_data
        self.logs = []

    def setup_trainer(self):
        """Configure SFTTrainer with training arguments"""

        training_args = TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=Config.LEARNING_RATE,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            report_to="none"  # Disable wandb for simplicity
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dataset_text_field="text"  # Will format in preprocessing
        )

        return trainer

    def train(self):
        """Execute training loop"""
        print("\n🏋️ Starting training...")

        # Preprocess data
        processor = GSM8KDataProcessor(self.tokenizer)

        # Format datasets
        def format_dataset(examples):
            texts = []
            for q, a in zip(examples['question'], examples['answer']):
                formatted = processor.format_example({"question": q, "answer": a})
                texts.append(formatted)
            return {"text": texts}

        train_formatted = self.train_data.map(format_dataset, batched=True)
        eval_formatted = self.eval_data.map(format_dataset, batched=True)

        # Setup trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            args=TrainingArguments(
                output_dir=Config.OUTPUT_DIR,
                num_train_epochs=Config.NUM_EPOCHS,
                per_device_train_batch_size=Config.BATCH_SIZE,
                gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
                learning_rate=Config.LEARNING_RATE,
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
                eval_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                fp16=True,
                optim="paged_adamw_8bit",
                weight_decay=0.01,
                report_to="none"
            ),
            tokenizer=self.tokenizer,
            max_seq_length=Config.MAX_SEQ_LENGTH
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(Config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(Config.OUTPUT_DIR)

        print(f"\n💾 Model saved to {Config.OUTPUT_DIR}")

        return trainer

    def evaluate_exact_match(self, trainer):
        """Evaluate using exact match accuracy"""
        print("\n📊 Running evaluation...")

        correct = 0
        total = 0
        results = []

        # Use a subset for quick evaluation
        eval_subset = self.eval_data.select(range(min(100, len(self.eval_data))))

        for example in tqdm(eval_subset, desc="Evaluating"):
            question = example['question']
            true_answer = example['answer'].split("####")[-1].strip()

            # Generate prediction
            prompt = f"### Problem:\n{question}\n\n### Solution:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract final answer from generation
            predicted_answer = self._extract_answer(generated)

            # Check exact match
            is_correct = self._normalize_answer(predicted_answer) == self._normalize_answer(true_answer)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question,
                'predicted': predicted_answer,
                'true': true_answer,
                'correct': is_correct
            })

        accuracy = correct / total if total > 0 else 0
        print(f"\n✅ Exact Match Accuracy: {accuracy:.2%} ({correct}/{total})")

        # Save evaluation results
        with open(os.path.join(Config.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump({
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'samples': results
            }, f, indent=2)

        return accuracy

    def _extract_answer(self, text):
        """Extract final numerical answer from generated text"""
        # Look for #### pattern
        if "####" in text:
            return text.split("####")[-1].strip().split()[0]

        # Look for "Final Answer:" pattern
        if "Final Answer:" in text:
            match = re.search(r"Final Answer:\s*([^\s]+)", text)
            if match:
                return match.group(1)

        # Extract last number
        numbers = re.findall(r"\d+", text)
        return numbers[-1] if numbers else ""

    def _normalize_answer(self, answer):
        """Normalize answer for comparison"""
        # Remove commas, dollar signs, etc.
        answer = answer.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            # Convert to float for numerical comparison
            return str(float(answer))
        except:
            return answer.lower().strip()

def main():
    """Main execution pipeline"""
    print("="*60)
    print("GSM8K Training with LLaMA 3.2 1B + LoRA")
    print("="*60)

    # Setup model
    model_setup = LoRAModelSetup()
    model, tokenizer = model_setup.setup()

    # Load data
    data_processor = GSM8KDataProcessor(tokenizer)
    train_data, eval_data = data_processor.load_and_split()

    # Initialize trainer
    trainer_wrapper = GSM8KTrainer(model, tokenizer, train_data, eval_data)

    # Train
    trainer = trainer_wrapper.train()

    # Evaluate
    accuracy = trainer_wrapper.evaluate_exact_match(trainer)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Model saved to: {Config.OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
