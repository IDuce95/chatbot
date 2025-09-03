import pytest
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.rag_manager import RAGManager


class TestRAGSystem:
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for test database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_chroma.db")

    @pytest.fixture
    def mock_config(self):
        return {
            "rag": {
                "collection_name": "test_collection",
                "persist_directory": "test_db",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_results": 5,
                "similarity_threshold": 0.5
            },
            "embeddings": {
                "model": "text-embedding-ada-002",
                "batch_size": 100
            }
        }

    @pytest.fixture
    def rag_manager(self, mock_config, temp_db_path):
        """Create RAGManager instance for testing"""
        # Update config with temp directory
        mock_config["rag"]["persist_directory"] = temp_db_path
        return RAGManager(mock_config)

    def test_rag_manager_initialization(self, rag_manager, mock_config):
        """Test RAGManager initializes correctly"""
        assert rag_manager.config == mock_config
        assert rag_manager.collection_name == "test_collection"
        assert rag_manager.chunk_size == 1000
        assert rag_manager.chunk_overlap == 200

    def test_rag_manager_document_splitting(self, rag_manager):
        """Test document text splitting functionality"""
        # Test with sample text
        sample_text = "This is a test document. " * 200  # Create long text

        chunks = rag_manager._split_text(sample_text)

        assert len(chunks) > 1  # Should split long text
        assert all(len(chunk) <= rag_manager.chunk_size + rag_manager.chunk_overlap for chunk in chunks)

    def test_rag_manager_document_processing(self, rag_manager):
        """Test document processing pipeline"""
        # Test document preparation
        documents = [
            {"content": "First document content", "source": "doc1.pdf"},
            {"content": "Second document content", "source": "doc2.pdf"}
        ]

        processed = rag_manager._prepare_documents(documents)

        assert len(processed) == 2
        assert all("content" in doc for doc in processed)
        assert all("metadata" in doc for doc in processed)

    def test_rag_manager_similarity_search_formatting(self, rag_manager):
        """Test search result formatting"""
        # Mock search results
        mock_results = [
            type('Result', (), {
                'page_content': 'Test content 1',
                'metadata': {'source': 'doc1.pdf', 'page': 1}
            })(),
            type('Result', (), {
                'page_content': 'Test content 2',
                'metadata': {'source': 'doc2.pdf', 'page': 2}
            })()
        ]

        formatted = rag_manager._format_search_results(mock_results, "test query")

        assert len(formatted) == 2
        assert all("content" in result for result in formatted)
        assert all("source" in result for result in formatted)
        assert all("relevance_score" in result for result in formatted)

    def test_rag_manager_query_processing(self, rag_manager):
        """Test query preprocessing"""
        # Test query cleaning
        queries = [
            "What is LangChain?",
            "How to implement    agents?",
            "Code examples!!!",
            ""
        ]

        for query in queries:
            processed = rag_manager._preprocess_query(query)

            if query.strip():  # Non-empty queries
                assert isinstance(processed, str)
                assert len(processed.strip()) > 0
            else:  # Empty queries
                assert processed == query

    def test_rag_manager_metadata_extraction(self, rag_manager):
        """Test metadata extraction from documents"""
        # Test file path metadata extraction
        file_paths = [
            "docs/langchain_tutorial.pdf",
            "examples/code_samples.md",
            "guides/installation.txt"
        ]

        for path in file_paths:
            metadata = rag_manager._extract_metadata(path)

            assert "source" in metadata
            assert "file_type" in metadata
            assert metadata["source"] == os.path.basename(path)

    def test_rag_manager_result_filtering(self, rag_manager):
        """Test search result filtering by relevance"""
        # Mock results with different relevance scores
        results = [
            {"content": "High relevance", "relevance_score": 0.9, "source": "doc1"},
            {"content": "Medium relevance", "relevance_score": 0.6, "source": "doc2"},
            {"content": "Low relevance", "relevance_score": 0.3, "source": "doc3"},
            {"content": "Very low relevance", "relevance_score": 0.1, "source": "doc4"}
        ]

        filtered = rag_manager._filter_results_by_relevance(results, threshold=0.5)

        assert len(filtered) == 2  # Should keep only high and medium relevance
        assert all(result["relevance_score"] >= 0.5 for result in filtered)

    def test_rag_manager_search_optimization(self, rag_manager):
        """Test search query optimization"""
        # Test different query types
        queries = [
            "langchain agents implementation",
            "how to use vector databases",
            "code examples python",
            "documentation setup guide"
        ]

        for query in queries:
            optimized = rag_manager._optimize_search_query(query)

            assert isinstance(optimized, str)
            assert len(optimized) > 0
            # Should preserve key terms
            for word in query.split():
                if len(word) > 3:  # Preserve meaningful words
                    assert word.lower() in optimized.lower()

    def test_rag_manager_batch_processing(self, rag_manager):
        """Test batch document processing"""
        # Test batch document handling
        documents = [
            {"content": f"Document {i} content", "source": f"doc{i}.pdf"}
            for i in range(10)
        ]

        batches = rag_manager._create_document_batches(documents, batch_size=3)

        assert len(batches) == 4  # 10 documents in batches of 3
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # Last batch has remainder

    def test_rag_manager_error_handling(self, rag_manager):
        """Test RAGManager error handling"""
        # Test with invalid inputs
        invalid_inputs = [
            None,
            [],
            "",
            {"invalid": "format"}
        ]

        for invalid_input in invalid_inputs:
            try:
                # Should handle gracefully
                result = rag_manager._safe_process(invalid_input)
                assert result is not None or result == []
            except Exception as e:
                # Should not raise unhandled exceptions
                assert isinstance(e, (ValueError, TypeError))

    def test_rag_manager_search_result_ranking(self, rag_manager):
        """Test search result ranking and scoring"""
        # Mock search results
        results = [
            {"content": "Python code example", "source": "python_guide.pdf"},
            {"content": "Java code example", "source": "java_guide.pdf"},
            {"content": "JavaScript example", "source": "js_guide.pdf"}
        ]

        # Test ranking for Python-specific query
        query = "Python programming examples"
        ranked = rag_manager._rank_results(results, query)

        assert len(ranked) == len(results)
        assert all("relevance_score" in result for result in ranked)
        # Python result should have higher score for Python query
        python_result = next(r for r in ranked if "Python" in r["content"])
        assert python_result["relevance_score"] > 0

    def test_rag_manager_content_deduplication(self, rag_manager):
        """Test content deduplication"""
        # Test with duplicate content
        documents = [
            {"content": "Duplicate content", "source": "doc1.pdf"},
            {"content": "Duplicate content", "source": "doc2.pdf"},
            {"content": "Unique content", "source": "doc3.pdf"},
            {"content": "Another unique content", "source": "doc4.pdf"}
        ]

        deduplicated = rag_manager._deduplicate_content(documents)

        # Should remove exact duplicates
        assert len(deduplicated) == 3
        contents = [doc["content"] for doc in deduplicated]
        assert len(set(contents)) == len(contents)  # All unique

    def test_rag_manager_search_performance(self, rag_manager):
        """Test search performance with different query sizes"""
        import time

        queries = [
            "short",
            "medium length query about programming",
            "very long query about programming concepts and implementation details that spans multiple topics and requires comprehensive search"
        ]

        for query in queries:
            start_time = time.time()

            # Simulate search preprocessing
            processed = rag_manager._preprocess_query(query)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should process quickly (under 1 second)
            assert processing_time < 1.0
            assert processed is not None

    def test_rag_manager_configuration_validation(self, rag_manager, mock_config):
        """Test configuration validation"""
        # Test valid configuration
        assert rag_manager._validate_config(mock_config)

        # Test invalid configurations
        invalid_configs = [
            {},  # Empty config
            {"rag": {}},  # Missing required fields
            {"rag": {"chunk_size": -1}},  # Invalid values
            {"rag": {"collection_name": ""}},  # Empty required field
        ]

        for invalid_config in invalid_configs:
            assert not rag_manager._validate_config(invalid_config)
