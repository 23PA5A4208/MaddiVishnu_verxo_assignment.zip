"""
Document Ingestion System with Sliding Window + Knowledge Pyramid
Vexoo Labs AI Engineer Assignment - Part 1
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class PyramidLevel:
    """Represents one level of the Knowledge Pyramid"""
    raw_text: str
    chunk_summary: str
    category: str
    distilled_knowledge: Dict

class SlidingWindowProcessor:
    """Implements 2-page sliding window strategy"""

    def __init__(self, window_size: int = 2500, overlap: int = 500):
        self.window_size = window_size
        self.overlap = overlap

    def create_windows(self, text: str) -> List[str]:
        """Create overlapping sliding windows from text"""
        windows = []
        start = 0

        while start < len(text):
            end = min(start + self.window_size, len(text))
            window = text[start:end]
            windows.append(window)

            start += (self.window_size - self.overlap)

            if end == len(text):
                break

        return windows

class KnowledgePyramidBuilder:
    """Builds hierarchical knowledge pyramid from text chunks"""

    def __init__(self):
        self.categories = {
            'technical': ['code', 'algorithm', 'system', 'api', 'database', 'function', 'class'],
            'business': ['revenue', 'profit', 'market', 'customer', 'sales', 'strategy', 'growth'],
            'legal': ['contract', 'agreement', 'clause', 'liability', 'compliance', 'regulation'],
            'general': ['introduction', 'overview', 'summary', 'conclusion']
        }

    def generate_summary(self, text: str, num_sentences: int = 2) -> str:
        """Placeholder summarization - extract first N sentences"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return '. '.join(sentences[:num_sentences]) + '.' if sentences else text[:100]

    def classify_category(self, text: str) -> str:
        """Rule-based category classification"""
        text_lower = text.lower()
        scores = {}

        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score

        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'

    def distill_knowledge(self, text: str) -> Dict:
        """Create distilled knowledge - keywords and mock embedding"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        embedding = np.random.randn(128)
        embedding = embedding / np.linalg.norm(embedding)

        return {
            'keywords': [kw[0] for kw in keywords],
            'embedding': embedding.tolist(),
            'word_count': len(words),
            'char_count': len(text)
        }

    def build_pyramid(self, window_text: str) -> PyramidLevel:
        """Build complete pyramid for a single window"""
        return PyramidLevel(
            raw_text=window_text,
            chunk_summary=self.generate_summary(window_text),
            category=self.classify_category(window_text),
            distilled_knowledge=self.distill_knowledge(window_text)
        )

class SemanticRetriever:
    """Retrieves relevant information using semantic similarity"""

    def __init__(self):
        self.pyramid_levels: List[PyramidLevel] = []

    def add_pyramid(self, pyramid: PyramidLevel):
        """Add pyramid level to index"""
        self.pyramid_levels.append(pyramid)

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple fuzzy text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant responses from any pyramid level"""
        scores = []

        for idx, pyramid in enumerate(self.pyramid_levels):
            raw_sim = self.text_similarity(query, pyramid.raw_text) * 0.3
            summary_sim = self.text_similarity(query, pyramid.chunk_summary) * 0.4
            category_sim = 1.0 if pyramid.category in query.lower() else 0.0

            query_words = set(query.lower().split())
            keyword_match = len(query_words.intersection(set(pyramid.distilled_knowledge['keywords'])))
            keyword_sim = keyword_match / max(len(query_words), 1) * 0.3

            total_score = raw_sim + summary_sim + category_sim + keyword_sim

            scores.append({
                'index': idx,
                'score': total_score,
                'pyramid': pyramid,
                'best_match_level': self._determine_best_level(query, pyramid, raw_sim, summary_sim, keyword_sim)
            })

        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def _determine_best_level(self, query: str, pyramid: PyramidLevel, 
                             raw_sim: float, summary_sim: float, keyword_sim: float) -> str:
        """Determine which pyramid level is most relevant"""
        levels = {
            'raw_text': raw_sim,
            'chunk_summary': summary_sim,
            'distilled_keywords': keyword_sim
        }
        return max(levels, key=levels.get)

class DocumentIngestionSystem:
    """Main system orchestrating the complete pipeline"""

    def __init__(self, window_size: int = 2500, overlap: int = 500):
        self.window_processor = SlidingWindowProcessor(window_size, overlap)
        self.pyramid_builder = KnowledgePyramidBuilder()
        self.retriever = SemanticRetriever()

    def ingest_document(self, text: str) -> List[PyramidLevel]:
        """Complete ingestion pipeline"""
        print(f"Ingesting document ({len(text)} characters)...")

        windows = self.window_processor.create_windows(text)
        print(f"Created {len(windows)} sliding windows")

        pyramids = []
        for i, window in enumerate(windows):
            pyramid = self.pyramid_builder.build_pyramid(window)
            pyramids.append(pyramid)
            self.retriever.add_pyramid(pyramid)
            print(f"  Pyramid {i+1}: Category='{pyramid.category}', Keywords={len(pyramid.distilled_knowledge['keywords'])}")

        print(f"Ingestion complete. Indexed {len(pyramids)} pyramid levels.")
        return pyramids

    def query(self, query_text: str, top_k: int = 3) -> Dict:
        """Query the system and retrieve relevant information"""
        print(f"Query: '{query_text}'")
        results = self.retriever.retrieve(query_text, top_k)

        response = {
            'query': query_text,
            'results': []
        }

        for i, result in enumerate(results):
            pyramid = result['pyramid']
            response['results'].append({
                'rank': i + 1,
                'relevance_score': round(result['score'], 4),
                'best_match_level': result['best_match_level'],
                'category': pyramid.category,
                'summary': pyramid.chunk_summary[:200] + "..." if len(pyramid.chunk_summary) > 200 else pyramid.chunk_summary,
                'keywords': pyramid.distilled_knowledge['keywords'][:5],
                'raw_preview': pyramid.raw_text[:150] + "..." if len(pyramid.raw_text) > 150 else pyramid.raw_text
            })

        return response

    def save_index(self, filepath: str):
        """Save the knowledge pyramid index to JSON"""
        data = []
        for pyramid in self.retriever.pyramid_levels:
            pyramid_dict = asdict(pyramid)
            pyramid_dict['distilled_knowledge']['embedding'] = pyramid.distilled_knowledge['embedding'][:10] + ['...']
            data.append(pyramid_dict)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Index saved to {filepath}")
