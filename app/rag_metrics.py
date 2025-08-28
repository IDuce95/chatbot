import time
from typing import List, Dict, Any
import numpy as np
import json
import math
from collections import Counter
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class RAGMetrics:
    def __init__(self):
        self.retrieval_metrics = {}
        self.generation_metrics = {}
        self.session_metrics = []

    def calculate_recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        if not relevant_docs:
            return 0.0

        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)

    def calculate_precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        if not retrieved_docs or k == 0:
            return 0.0

        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
        return relevant_retrieved / min(k, len(retrieved_k))

    def calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_ndcg_at_k(self, retrieved_docs: List[str], relevance_scores: List[float], k: int) -> float:
        if not retrieved_docs or not relevance_scores:
            return 0.0

        k = min(k, len(retrieved_docs), len(relevance_scores))

        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(relevance_scores[:k]))

        sorted_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(sorted_scores[:k]))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], ground_truth: List[str] = None, k_values: List[int] = None) -> Dict[str, float]:
        if k_values is None:
            k_values = [1, 3, 5, 10]

        retrieved_ids = [doc.get('id', str(i)) for i, doc in enumerate(retrieved_docs)]
        scores = [doc.get('score', 1.0) for doc in retrieved_docs]

        basic_metrics = {
            'total_retrieved': len(retrieved_docs),
            'query_length': len(query.split()),
            'avg_score': np.mean(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0
        }

        if ground_truth:
            for k in k_values:
                basic_metrics[f'recall@{k}'] = self.calculate_recall_at_k(retrieved_ids, ground_truth, k)
                basic_metrics[f'precision@{k}'] = self.calculate_precision_at_k(retrieved_ids, ground_truth, k)

            basic_metrics['mrr'] = self.calculate_mrr(retrieved_ids, ground_truth)

            relevance_scores = [1.0 if doc_id in ground_truth else 0.0 for doc_id in retrieved_ids]
            for k in k_values:
                basic_metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(retrieved_ids, relevance_scores, k)

        auto_metrics = self.evaluate_retrieval_with_auto_relevance(query, retrieved_docs, k_values)
        basic_metrics.update(auto_metrics)

        return basic_metrics

    def calculate_bleu_score(self, reference: str, candidate: str) -> Dict[str, float]:
        if not NLTK_AVAILABLE:
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

        reference_tokens = [reference.lower().split()]
        candidate_tokens = candidate.lower().split()

        smoothing = SmoothingFunction().method1

        try:
            bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            return {
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4
            }
        except Exception:
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        if not NLTK_AVAILABLE:
            return {'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0}

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)

            return {
                'rouge_1_f': scores['rouge1'].fmeasure,
                'rouge_1_p': scores['rouge1'].precision,
                'rouge_1_r': scores['rouge1'].recall,
                'rouge_2_f': scores['rouge2'].fmeasure,
                'rouge_2_p': scores['rouge2'].precision,
                'rouge_2_r': scores['rouge2'].recall,
                'rouge_l_f': scores['rougeL'].fmeasure,
                'rouge_l_p': scores['rougeL'].precision,
                'rouge_l_r': scores['rougeL'].recall
            }
        except Exception:
            return {'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0}

    def calculate_perplexity_approx(self, text: str) -> float:
        if not text:
            return float('inf')

        words = text.lower().split()
        if len(words) < 2:
            return 1.0

        word_counts = Counter(words)
        total_words = len(words)

        log_prob_sum = 0.0
        for word in words:
            prob = word_counts[word] / total_words
            log_prob_sum += math.log(prob)

        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)

        return perplexity

    def calculate_repetition_metrics(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        if not words:
            return {'repetition_ratio': 0.0, 'unique_word_ratio': 0.0}

        unique_words = set(words)
        repetition_ratio = 1.0 - (len(unique_words) / len(words))
        unique_word_ratio = len(unique_words) / len(words)

        return {
            'repetition_ratio': repetition_ratio,
            'unique_word_ratio': unique_word_ratio,
            'total_words': len(words),
            'unique_words': len(unique_words)
        }

    def calculate_semantic_relevance(self, query: str, document_content: str) -> float:
        query_words = set(query.lower().split())
        doc_words = set(document_content.lower().split())

        if not query_words or not doc_words:
            return 0.0

        overlap = query_words & doc_words

        jaccard_similarity = len(overlap) / len(query_words | doc_words)

        keyword_match_ratio = len(overlap) / len(query_words)

        combined_score = (jaccard_similarity + keyword_match_ratio) / 2.0

        boost_words = {'langchain', 'framework', 'python', 'programming', 'code', 'api', 'library'}
        boost_overlap = len(overlap & boost_words)
        if boost_overlap > 0:
            combined_score += boost_overlap * 0.1

        return min(combined_score, 1.0)

    def calculate_distance_based_relevance(self, distance: float, max_distance: float = 0.4) -> float:
        if distance >= max_distance:
            return 0.0

        relevance = 1.0 - (distance / max_distance)
        return relevance

    def determine_automatic_ground_truth(self, query: str, retrieved_docs: List[Dict], relevance_threshold: float = 0.6) -> List[str]:
        ground_truth = []

        for doc in retrieved_docs:
            doc_id = doc.get('id', 'unknown')
            content = doc.get('content', '')
            distance = doc.get('distance', 1.0)

            semantic_relevance = self.calculate_semantic_relevance(query, content)
            distance_relevance = self.calculate_distance_based_relevance(distance)

            combined_relevance = (semantic_relevance + distance_relevance) / 2.0

            if combined_relevance >= relevance_threshold:
                ground_truth.append(doc_id)

        return ground_truth

    def evaluate_retrieval_with_auto_relevance(self, query: str, retrieved_docs: List[Dict], k_values: List[int] = None, relevance_threshold: float = 0.6) -> Dict[str, Any]:
        if k_values is None:
            k_values = [1, 3, 5, 10]

        retrieved_ids = [doc.get('id', str(i)) for i, doc in enumerate(retrieved_docs)]
        scores = [doc.get('score', 1.0) for doc in retrieved_docs]
        distances = [doc.get('distance', 1.0) for doc in retrieved_docs]

        metrics = {
            'total_retrieved': len(retrieved_docs),
            'query_length': len(query.split()),
            'avg_score': np.mean(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'avg_distance': np.mean(distances) if distances else 0.0,
            'min_distance': min(distances) if distances else 0.0
        }

        auto_ground_truth = self.determine_automatic_ground_truth(query, retrieved_docs, relevance_threshold)

        metrics['auto_relevant_count'] = len(auto_ground_truth)
        metrics['auto_relevance_rate'] = len(auto_ground_truth) / len(retrieved_docs) if retrieved_docs else 0.0

        if auto_ground_truth:
            for k in k_values:
                metrics[f'auto_recall@{k}'] = self.calculate_recall_at_k(retrieved_ids, auto_ground_truth, k)
                metrics[f'auto_precision@{k}'] = self.calculate_precision_at_k(retrieved_ids, auto_ground_truth, k)

            metrics['auto_mrr'] = self.calculate_mrr(retrieved_ids, auto_ground_truth)

            auto_relevance_scores = []
            for doc in retrieved_docs:
                content = doc.get('content', '')
                distance = doc.get('distance', 1.0)
                semantic_rel = self.calculate_semantic_relevance(query, content)
                distance_rel = self.calculate_distance_based_relevance(distance)
                combined_rel = (semantic_rel + distance_rel) / 2.0
                auto_relevance_scores.append(combined_rel)

            for k in k_values:
                metrics[f'auto_ndcg@{k}'] = self.calculate_ndcg_at_k(retrieved_ids, auto_relevance_scores, k)

        return metrics

    def calculate_response_time(self, start_time: float, end_time: float) -> float:
        return end_time - start_time

    def calculate_context_utilization(self, context: str, response: str) -> Dict[str, float]:
        if not context or not response:
            return {'utilization_ratio': 0.0, 'context_coverage': 0.0}

        context_words = set(context.lower().split())
        response_words = set(response.lower().split())

        if not context_words:
            return {'utilization_ratio': 0.0, 'context_coverage': 0.0}

        overlap = context_words & response_words
        utilization_ratio = len(overlap) / len(context_words)
        context_coverage = len(overlap) / len(response_words) if response_words else 0.0

        return {
            'utilization_ratio': utilization_ratio,
            'context_coverage': context_coverage,
            'overlap_words': len(overlap)
        }

    def evaluate_generation(self, query: str, response: str, context: str = "", reference: str = "") -> Dict[str, Any]:
        metrics = {
            'response_length': len(response),
            'response_word_count': len(response.split()),
            'query_length': len(query),
            'context_length': len(context)
        }

        if context:
            context_metrics = self.calculate_context_utilization(context, response)
            metrics.update(context_metrics)

        if reference:
            metrics['reference_similarity'] = self._calculate_simple_similarity(response, reference)

            bleu_scores = self.calculate_bleu_score(reference, response)
            metrics.update(bleu_scores)

            rouge_scores = self.calculate_rouge_scores(reference, response)
            metrics.update(rouge_scores)

        repetition_metrics = self.calculate_repetition_metrics(response)
        metrics.update(repetition_metrics)

        metrics['perplexity_approx'] = self.calculate_perplexity_approx(response)

        return metrics

    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def log_interaction(self, query: str, retrieved_docs: List[Dict], response: str, context: str = "", response_time: float = 0.0, rag_used: bool = False, quality_score: float = 0.0):
        try:
            interaction = {
                'timestamp': time.time(),
                'query': query,
                'response': response,
                'context': context,
                'response_time': response_time,
                'rag_used': rag_used,
                'retrieved_docs_count': len(retrieved_docs) if retrieved_docs else 0,
                'retrieval_metrics': self.evaluate_retrieval(query, retrieved_docs),
                'generation_metrics': self.evaluate_generation(query, response, context),
                'quality_score': quality_score
            }

            self.session_metrics.append(interaction)
            return interaction
        except Exception as e:
            basic_interaction = {
                'timestamp': time.time(),
                'query': query,
                'response': response,
                'context': context,
                'response_time': response_time,
                'rag_used': rag_used,
                'retrieved_docs_count': len(retrieved_docs) if retrieved_docs else 0,
                'retrieval_metrics': {},
                'generation_metrics': {}
            }
            self.session_metrics.append(basic_interaction)
            logging.warning(f"Warning: log_interaction error: {e}")
            return basic_interaction

    def get_session_summary(self) -> Dict[str, Any]:
        if not self.session_metrics:
            return {'total_interactions': 0}

        total_interactions = len(self.session_metrics)
        rag_interactions = sum(1 for m in self.session_metrics if m['rag_used'])

        avg_response_time = np.mean([m['response_time'] for m in self.session_metrics])
        avg_retrieved_docs = np.mean([m['retrieved_docs_count'] for m in self.session_metrics])

        return {
            'total_interactions': total_interactions,
            'rag_interactions': rag_interactions,
            'rag_usage_rate': rag_interactions / total_interactions,
            'avg_response_time': avg_response_time,
            'avg_retrieved_docs': avg_retrieved_docs,
            'total_session_time': time.time() - self.session_metrics[0]['timestamp'] if self.session_metrics else 0
        }

    def export_metrics(self, filepath: str):
        data = {
            'session_summary': self.get_session_summary(),
            'interactions': self.session_metrics
        }

        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        data = convert_numpy_types(data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def clear_session(self):
        self.session_metrics = []
