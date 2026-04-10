# Vexoo Labs AI Engineer Assignment

Complete implementation of document ingestion system with sliding window + knowledge pyramid, GSM8K training with LLaMA 3.2 1B, and reasoning-aware adapter.

## 📁 Project Structure

```
.
├── document_ingestion.py      # Part 1: Sliding window + Knowledge pyramid
├── gsm8k_train.py            # Part 2: LoRA fine-tuning on GSM8K
├── reasoning_adapter.py       # Bonus: Dynamic reasoning router
├── main.py                    # Demo and integration script
├── README.md                  # This file
├── requirements.txt           # Dependencies
└── output/                    # Generated outputs
    ├── knowledge_index.json   # Pyramid index
    └── query_results.json     # Query results
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Document Ingestion Demo

```bash
python main.py
```

This will:
1. Process a sample document using sliding windows
2. Build knowledge pyramids for each chunk
3. Execute sample queries and show retrieval results
4. Demonstrate the reasoning adapter

### Run GSM8K Training

```bash
python gsm8k_train.py
```

**Note:** Requires HuggingFace token with LLaMA 3.2 access and GPU with 16GB+ VRAM (uses 4-bit quantization).

## 📚 Part 1: Document Ingestion System

### Sliding Window Strategy

- **Window Size**: 800-2500 characters (configurable)
- **Overlap**: 100-500 characters to preserve context
- **Implementation**: Character-based with sentence boundary awareness

### Knowledge Pyramid (4 Layers)

| Layer | Description | Implementation |
|-------|-------------|----------------|
| Raw Text | Original content | Direct storage |
| Chunk Summary | Brief overview | First N sentences extraction |
| Category/Theme | Content classification | Rule-based keyword matching |
| Distilled Knowledge | Compact representation | Top keywords + mock embeddings |

### Retrieval

- Multi-level semantic similarity
- Fuzzy matching across pyramid layers
- Returns best matching level with confidence score

## 🧠 Part 2: GSM8K Training

### Configuration

- **Model**: meta-llama/Llama-3.2-1B-Instruct
- **Dataset**: openai/gsm8k (3000 train, 1000 eval)
- **Method**: LoRA (r=16, alpha=32) + 4-bit quantization
- **Training**: 3 epochs, batch size 4, learning rate 2e-4

### Features

- Instruction-following format for math problems
- Chain-of-thought prompting
- Exact match accuracy evaluation
- Automatic answer extraction from generated text

## 🔌 Bonus: Reasoning-Aware Adapter

### Architecture

```
Query → Classifier → Router → Specialized Module → Result
              ↓
    [Math | Legal | Code | General]
```

### Key Components

1. **QuestionClassifier**: Rule-based type detection
2. **ReasoningRouter**: Dynamic module selection
3. **ReasoningModule**: Pluggable base class for specialized reasoning

### Extending

```python
class CustomReasoning(ReasoningModule):
    def can_handle(self, context):
        return context.question_type == QuestionType.CUSTOM

    def reason(self, context):
        return {'approach': 'custom_logic', ...}

router.register_module(CustomReasoning())
```

## 🛠️ Design Decisions

1. **Character-based sliding window**: Faster than token-based, language agnostic
2. **Placeholder summarization**: Focus on architecture; can swap in BART/T5 later
3. **Mock embeddings**: Demonstrates pyramid structure; production would use real vectors
4. **LoRA over full fine-tuning**: Memory efficient, prevents catastrophic forgetting
5. **4-bit quantization**: Enables 1B model training on consumer GPUs

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers, peft, trl, datasets
- numpy, tqdm

See `requirements.txt` for complete list.

## 📝 Notes

- GSM8K training requires HuggingFace authentication
- First run downloads ~2GB model weights
- Evaluation uses exact match; partial credit not implemented
- Reasoning adapter uses rule-based classification (ML classifier can be added)

## 📧 Contact

For questions about this implementation, refer to the inline documentation and comments.
