from unittest.mock import Mock

import pytest
from langchain.schema import Document

from app.rag_manager import RAGManager


class TestRAG:

    def test_split_documents_with_mock(self):
        mock_doc1 = Mock(spec=Document)
        mock_doc1.page_content = "This is test content for document one. " * 100
        mock_doc1.metadata = {"source": "test1.pdf", "page": 1}

        mock_doc2 = Mock(spec=Document)
        mock_doc2.page_content = "This is different content for document two. " * 100
        mock_doc2.metadata = {"source": "test2.pdf", "page": 1}

        documents = [mock_doc1, mock_doc2]

        try:
            rag = RAGManager("config.toml")
            chunks = rag.split_documents(documents)
            assert len(chunks) >= 2

            for chunk in chunks:
                assert hasattr(chunk, 'page_content')
                assert hasattr(chunk, 'metadata')
                assert len(chunk.page_content) > 0

        except FileNotFoundError:
            pytest.skip("config.toml not found - skipping integration test")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    def test_mock_document_creation(self):
        good_mock = Mock(spec=Document)
        good_mock.page_content = "Test content"
        good_mock.metadata = {"source": "test.pdf"}

        assert good_mock.page_content == "Test content"
        assert good_mock.metadata["source"] == "test.pdf"

        assert hasattr(good_mock, 'page_content')
        assert hasattr(good_mock, 'metadata')

    def test_load_pdf_documents_method_exists(self):
        try:
            rag = RAGManager("config.toml")
            assert hasattr(rag, 'load_pdf_documents')
            assert hasattr(rag, 'split_documents')
            assert hasattr(rag, 'search_documents')
            assert hasattr(rag, 'get_context_with_relevance')

        except FileNotFoundError:
            pytest.skip("config.toml not found - skipping test")
