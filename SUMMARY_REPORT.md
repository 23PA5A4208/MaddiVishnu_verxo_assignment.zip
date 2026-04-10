# Vexoo Labs Assignment Summary

## Part 1: Document Ingestion + Knowledge Pyramid

**Sliding Window Strategy**
- Character-based windows (800-2500 chars) with 100-500 char overlap
- Overlaps preserve context across boundaries
- Configurable for different document types

**Knowledge Pyramid (4 Layers)**
| Layer | Content | Method |
|-------|---------|--------|
| Raw Text | Original chunk | Direct storage |
| Summary | Overview | First N sentences |
| Category | Classification | Rule-based keywords |
| Distilled | Keywords + embedding | Frequency + mock vectors |

**Retrieval**: Multi-level fuzzy matching across pyramid layers, returns best match with confidence score.

## Part 2: GSM8K Training

**Setup**
- Model: LLaMA 3.2 1B Instruct
- Dataset: 3000 train / 1000 eval from openai/gsm8k
- Method: LoRA (r=16, α=32) + 4-bit quantization

**Training Loop**
- 3 epochs, batch size 4, LR 2e-4
- Instruction-following format with chain-of-thought
- Evaluation: Exact match accuracy on numerical answers

## Key Design Decisions

1. **Character windows** > token windows (speed, simplicity)
2. **LoRA** > full fine-tuning (memory, no catastrophic forgetting)
3. **Modular architecture** for extensibility
4. **Rule-based classification** (easily upgraded to ML classifier)

## Bonus: Reasoning Adapter

Plug-and-play router with specialized modules:
- **Math**: Symbolic solver + step-by-step
- **Legal**: Structured retrieval + citation
- **Code**: Generation + syntax check
- **General**: Semantic search

Extensible via `ReasoningModule` base class.

---
**Architecture Focus**: Modular, scalable, production-ready patterns over quick hacks.
