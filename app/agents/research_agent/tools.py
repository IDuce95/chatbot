from typing import Any, Dict, List

from ..base_tool import BaseTool


class VectorDBRetrieverTool(BaseTool):
    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config

    def execute(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if hasattr(self.chatbot, 'rag_manager') and self.chatbot.rag_manager:
                results = self.chatbot.rag_manager.search_documents(query, top_k=top_k)

                formatted_results = []
                for doc, score in results:
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_score': score,
                        'source': doc.metadata.get('source', 'unknown')
                    })

                return formatted_results
            else:
                print("⚠️ RAG manager not available")
                return []

        except Exception as e:
            print(f"Vector search error: {e}")
            return []


class KnowledgeFilterTool(BaseTool):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_relevance_score = config.get("agents", {}).get("research", {}).get("min_relevance", 0.3)

    def execute(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not documents:
            return []

        filtered_docs = [
            doc for doc in documents
            if doc.get('relevance_score', 0) >= self.min_relevance_score
        ]

        filtered_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        for i, doc in enumerate(filtered_docs):
            doc['rank'] = i + 1

        return filtered_docs
