"""
Bonus: Reasoning-Aware Adapter Architecture
Vexoo Labs AI Engineer Assignment

This module demonstrates a plug-and-play adapter design that dynamically
reasons about different types of questions based on input characteristics.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any
import re

class QuestionType(Enum):
    """Enumeration of supported question types"""
    MATHEMATICAL = auto()
    LEGAL = auto()
    GENERAL_KNOWLEDGE = auto()
    LOGICAL_REASONING = auto()
    CODE_GENERATION = auto()
    UNKNOWN = auto()

@dataclass
class ReasoningContext:
    """Context object carrying reasoning metadata through the pipeline"""
    query: str
    question_type: QuestionType
    complexity_score: float
    required_tools: List[str]
    reasoning_depth: int  # How many reasoning steps needed
    confidence: float

class QuestionClassifier:
    """
    Rule-based and ML-enhanced classifier to detect question type.
    Can be extended with a trained classifier model.
    """

    # Keyword patterns for classification
    PATTERNS = {
        QuestionType.MATHEMATICAL: [
            r'\d+', r'calculate', r'solve', r'equation', r'math', 
            r'plus', r'minus', r'times', r'divided', r'percentage',
            r'algebra', r'geometry', r'compute', r'sum', r'product'
        ],
        QuestionType.LEGAL: [
            r'law', r'legal', r'contract', r'clause', r'liability',
            r'regulation', r'compliance', r'court', r'jurisdiction',
            r'plaintiff', r'defendant', r'statute', r'legislation'
        ],
        QuestionType.CODE_GENERATION: [
            r'code', r'function', r'program', r'script', r'algorithm',
            r'python', r'javascript', r'java', r'implement', r'class',
            r'method', r'api', r'debug', r'refactor'
        ],
        QuestionType.LOGICAL_REASONING: [
            r'if.*then', r'logic', r'deduce', r'infer', r'conclusion',
            r'premise', r'syllogism', r'pattern', r'sequence'
        ]
    }

    def classify(self, query: str) -> QuestionType:
        """Classify query into question type using pattern matching"""
        query_lower = query.lower()
        scores = {qt: 0 for qt in QuestionType}

        for qtype, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[qtype] += 1

        # Return type with highest score, or UNKNOWN
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0 else QuestionType.GENERAL_KNOWLEDGE

class ReasoningModule:
    """Base class for specialized reasoning modules"""

    def __init__(self, name: str):
        self.name = name

    def can_handle(self, context: ReasoningContext) -> bool:
        """Check if this module can handle the given context"""
        raise NotImplementedError

    def reason(self, context: ReasoningContext) -> Dict[str, Any]:
        """Execute reasoning and return results"""
        raise NotImplementedError

class MathematicalReasoning(ReasoningModule):
    """Symbolic math reasoning with step-by-step solver"""

    def __init__(self):
        super().__init__("MathematicalReasoning")

    def can_handle(self, context: ReasoningContext) -> bool:
        return context.question_type == QuestionType.MATHEMATICAL

    def reason(self, context: ReasoningContext) -> Dict[str, Any]:
        """Simulated mathematical reasoning pipeline"""
        return {
            'approach': 'chain_of_thought',
            'steps': [
                'identify_variables',
                'formulate_equations', 
                'solve_step_by_step',
                'verify_solution'
            ],
            'tools_needed': ['calculator', 'symbolic_solver'],
            'complexity': context.complexity_score
        }

class LegalReasoning(ReasoningModule):
    """Legal reasoning with structured retrieval and citation"""

    def __init__(self):
        super().__init__("LegalReasoning")

    def can_handle(self, context: ReasoningContext) -> bool:
        return context.question_type == QuestionType.LEGAL

    def reason(self, context: ReasoningContext) -> Dict[str, Any]:
        """Simulated legal reasoning pipeline"""
        return {
            'approach': 'structured_retrieval',
            'steps': [
                'identify_legal_domain',
                'retrieve_relevant_statutes',
                'check_precedents',
                'apply_reasoning',
                'generate_citation'
            ],
            'tools_needed': ['legal_database', 'citation_engine'],
            'citation_required': True
        }

class GeneralKnowledgeReasoning(ReasoningModule):
    """Semantic search and retrieval-based reasoning"""

    def __init__(self):
        super().__init__("GeneralKnowledgeReasoning")

    def can_handle(self, context: ReasoningContext) -> bool:
        return context.question_type == QuestionType.GENERAL_KNOWLEDGE

    def reason(self, context: ReasoningContext) -> Dict[str, Any]:
        """Simulated general knowledge reasoning"""
        return {
            'approach': 'semantic_retrieval',
            'steps': [
                'embed_query',
                'search_knowledge_base',
                'rank_results',
                'synthesize_answer'
            ],
            'tools_needed': ['vector_database', 'embedding_model']
        }

class CodeReasoning(ReasoningModule):
    """Code generation and analysis reasoning"""

    def __init__(self):
        super().__init__("CodeReasoning")

    def can_handle(self, context: ReasoningContext) -> bool:
        return context.question_type == QuestionType.CODE_GENERATION

    def reason(self, context: ReasoningContext) -> Dict[str, Any]:
        """Simulated code reasoning pipeline"""
        return {
            'approach': 'structured_generation',
            'steps': [
                'analyze_requirements',
                'design_algorithm',
                'generate_code',
                'syntax_check',
                'test_cases'
            ],
            'tools_needed': ['code_generator', 'linter', 'test_runner']
        }

class ReasoningRouter:
    """
    Central router that manages reasoning modules and routes queries.
    Implements the plug-and-play adapter pattern.
    """

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.modules: List[ReasoningModule] = []
        self.register_default_modules()

    def register_default_modules(self):
        """Register built-in reasoning modules"""
        self.modules = [
            MathematicalReasoning(),
            LegalReasoning(),
            CodeReasoning(),
            GeneralKnowledgeReasoning()  # Fallback
        ]

    def register_module(self, module: ReasoningModule):
        """Plug in a new reasoning module dynamically"""
        self.modules.append(module)
        print(f"🔌 Registered new module: {module.name}")

    def unregister_module(self, name: str):
        """Remove a reasoning module"""
        self.modules = [m for m in self.modules if m.name != name]

    def analyze_query(self, query: str) -> ReasoningContext:
        """Analyze query and create reasoning context"""
        qtype = self.classifier.classify(query)

        # Calculate complexity (simple heuristic)
        complexity = min(len(query.split()) / 20.0, 1.0)

        # Determine reasoning depth based on question type
        depth_map = {
            QuestionType.MATHEMATICAL: 4,
            QuestionType.LEGAL: 5,
            QuestionType.CODE_GENERATION: 4,
            QuestionType.LOGICAL_REASONING: 3,
            QuestionType.GENERAL_KNOWLEDGE: 2
        }

        return ReasoningContext(
            query=query,
            question_type=qtype,
            complexity_score=complexity,
            required_tools=[],
            reasoning_depth=depth_map.get(qtype, 2),
            confidence=0.0
        )

    def route(self, query: str) -> Dict[str, Any]:
        """
        Main routing function - the entry point for the reasoning adapter.
        Returns complete reasoning plan.
        """
        # Step 1: Analyze query
        context = self.analyze_query(query)

        # Step 2: Find appropriate module
        selected_module = None
        for module in self.modules:
            if module.can_handle(context):
                selected_module = module
                break

        # Step 3: If no specific module found, use general knowledge
        if selected_module is None:
            selected_module = GeneralKnowledgeReasoning()
            context.question_type = QuestionType.GENERAL_KNOWLEDGE

        # Step 4: Execute reasoning
        reasoning_plan = selected_module.reason(context)

        return {
            'query': query,
            'detected_type': context.question_type.name,
            'selected_module': selected_module.name,
            'complexity': context.complexity_score,
            'reasoning_plan': reasoning_plan,
            'estimated_steps': context.reasoning_depth
        }

# Demonstration
if __name__ == "__main__":
    print("="*60)
    print("Reasoning-Aware Adapter Demonstration")
    print("="*60)

    # Initialize router
    router = ReasoningRouter()

    # Test queries
    test_queries = [
        "What is the derivative of x^2 + 3x - 5?",
        "Explain the concept of consideration in contract law",
        "Write a Python function to reverse a linked list",
        "Who was the first president of the United States?",
        "If all A are B, and all B are C, what can we infer about A and C?"
    ]

    print("\n🔍 Testing Reasoning Router:")
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  → Type: {result['detected_type']}")
        print(f"  → Module: {result['selected_module']}")
        print(f"  → Approach: {result['reasoning_plan']['approach']}")
        print(f"  → Steps: {len(result['reasoning_plan']['steps'])}")
