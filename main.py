"""
Vexoo Labs AI Engineer Assignment - Complete Solution
Main execution script demonstrating both Part 1 and Part 2
"""

import json
import os
from document_ingestion import DocumentIngestionSystem, PyramidLevel
from reasoning_adapter import ReasoningRouter, QuestionClassifier

def demo_document_ingestion():
    """Demonstrate Part 1: Document Ingestion with Sliding Window + Knowledge Pyramid"""
    print("="*70)
    print("PART 1: DOCUMENT INGESTION SYSTEM")
    print("="*70)

    # Sample document (simulating a technical/business document)
    sample_document = """
    Technical Architecture Overview

    The system is built on a microservices architecture using Kubernetes for orchestration.
    Each service is containerized using Docker and deployed across multiple availability zones.
    The API gateway handles authentication using JWT tokens and routes requests to appropriate services.

    Database Layer
    We use PostgreSQL for transactional data and Redis for caching frequently accessed queries.
    Data is replicated across three nodes for high availability. Backup procedures run daily at midnight.

    Business Strategy 2024
    Our revenue model focuses on subscription-based SaaS offerings. Customer acquisition cost has 
    decreased by 15% through optimized marketing channels. The target market includes enterprise 
    clients in the fintech sector.

    Machine Learning Pipeline
    The ML pipeline uses Apache Spark for data processing and TensorFlow for model training.
    Models are versioned using MLflow and deployed via a CI/CD pipeline. Feature engineering is 
    automated using a custom ETL framework.

    Legal Compliance
    All data processing complies with GDPR regulations. User consent is obtained before data 
    collection. Data retention policies are strictly enforced with automated deletion schedules.
    Contract agreements with third-party vendors include strict liability clauses.
    """

    # Initialize system
    ingestion_system = DocumentIngestionSystem(window_size=800, overlap=100)

    # Ingest document
    pyramids = ingestion_system.ingest_document(sample_document)

    # Save index
    os.makedirs("output", exist_ok=True)
    ingestion_system.save_index("output/knowledge_index.json")

    # Query examples
    print("\n" + "-"*70)
    print("QUERY EXAMPLES")
    print("-"*70)

    queries = [
        "What database systems are used?",
        "Tell me about the revenue model",
        "What are the legal compliance requirements?",
        "How is machine learning implemented?"
    ]

    all_results = []
    for query in queries:
        result = ingestion_system.query(query, top_k=2)
        all_results.append(result)

        print(f"\n🔍 Query: '{query}'")
        for r in result['results']:
            print(f"   Rank {r['rank']}: [{r['category'].upper()}] Score={r['relevance_score']}")
            print(f"   → Best match from: {r['best_match_level']}")
            print(f"   → Summary: {r['summary'][:80]}...")
            print(f"   → Keywords: {', '.join(r['keywords'])}")

    # Save query results
    with open("output/query_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return ingestion_system

def demo_reasoning_adapter():
    """Demonstrate Bonus: Reasoning-Aware Adapter"""
    print("\n" + "="*70)
    print("BONUS: REASONING-AWARE ADAPTER")
    print("="*70)

    router = ReasoningRouter()

    test_cases = [
        ("Calculate the area of a circle with radius 5", "Mathematical"),
        ("What are the requirements for a valid contract?", "Legal"),
        ("Write a function to sort an array in Python", "Code"),
        ("Who invented the telephone?", "General Knowledge"),
        ("Solve: 2x + 5 = 15", "Mathematical")
    ]

    print("\n📋 Test Cases:")
    for query, expected_type in test_cases:
        result = router.route(query)
        match = "✅" if expected_type.upper() in result['detected_type'] else "❌"
        print(f"\n{match} Query: {query}")
        print(f"   Detected: {result['detected_type']}")
        print(f"   Module: {result['selected_module']}")
        print(f"   Complexity: {result['complexity']:.2f}")
        print(f"   Reasoning Steps: {result['estimated_steps']}")

def print_summary():
    """Print implementation summary"""
    print("\n" + "="*70)
    print("IMPLEMENTATION SUMMARY")
    print("="*70)

    summary = """
PART 1 - DOCUMENT INGESTION SYSTEM:
✅ Sliding Window Strategy
   - Character-based windows (800 chars) with 100 char overlap
   - Preserves context across window boundaries
   - Configurable window size and overlap

✅ Knowledge Pyramid (4 Layers)
   Layer 1: Raw Text - Original document content
   Layer 2: Chunk Summary - First N sentences extraction
   Layer 3: Category/Theme - Rule-based classification (technical/business/legal/general)
   Layer 4: Distilled Knowledge - Keywords + mock embeddings

✅ Retrieval Strategy
   - Multi-level semantic similarity matching
   - Fuzzy text matching across all pyramid levels
   - Returns best matching level with relevance scores

PART 2 - GSM8K TRAINING (gsm8k_train.py):
✅ Dataset: openai/gsm8k from Hugging Face
✅ Model: meta-llama/Llama-3.2-1B-Instruct
✅ Fine-tuning: LoRA (r=16, alpha=32) with 4-bit quantization
✅ Training: 3000 samples, 3 epochs, batch size 4
✅ Evaluation: Exact match accuracy on 1000 samples

BONUS - REASONING ADAPTER:
✅ Dynamic routing based on question type
✅ Plug-and-play module architecture
✅ Supports: Math, Legal, Code, General Knowledge
✅ Extensible design for adding new reasoning types
"""
    print(summary)

if __name__ == "__main__":
    # Run demonstrations
    demo_document_ingestion()
    demo_reasoning_adapter()
    print_summary()

    print("\n💾 All outputs saved to ./output/ directory")
    print("✅ Assignment complete!")
