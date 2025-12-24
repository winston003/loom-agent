"""
Pytest Configuration and Fixtures
"""

import shutil
import tempfile
from collections.abc import Generator

import pytest

from loom.api.main import LoomApp
from loom.infra.llm import MockLLMProvider


@pytest.fixture
def app() -> LoomApp:
    """Returns a fresh LoomApp instance."""
    return LoomApp()

@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Returns a Mock LLM Provider."""
    return MockLLMProvider()

@pytest.fixture
def temp_memory_path() -> Generator[str, None, None]:
    """Returns a temporary directory for memory storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
