"""Signal Hunter AI — unit tests for semantic embedding deduplication."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from signal_hunter.dedup.semantic import SemanticDeduplicator, _cosine_similarity


def test_cosine_similarity():
    """Verify that the cosine similarity calculation behaves correctly."""
    # Identical vectors -> similarity of 1.0
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert pytest.approx(_cosine_similarity(a, b)) == 1.0

    # Orthogonal vectors -> similarity of 0.0
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert pytest.approx(_cosine_similarity(a, c)) == 0.0

    # Opposite vectors -> similarity of -1.0
    d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert pytest.approx(_cosine_similarity(a, d)) == -1.0


@pytest.mark.asyncio
async def test_semantic_deduplication_flow():
    """Verify semantic lookup matches similar items and rolls the window correctly."""
    dedup = SemanticDeduplicator()

    # Mock the _embed helper to return deterministic vectors
    async def mock_embed(model, text):
        if "devin" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "cursor" in text.lower():
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    with patch("signal_hunter.dedup.semantic._embed", side_effect=mock_embed), \
         patch("signal_hunter.dedup.semantic._get_model", return_value=AsyncMock()):
        
        # 1. Search in empty window should return None
        match_empty = await dedup.find_similar("Devin AI has launched", "The first autonomous AI software engineer", threshold=0.8)
        assert match_empty is None

        # 2. Add Devin item to the rolling window
        await dedup.add_to_window(
            cluster_id="cluster-devin",
            title="Devin AI has launched",
            snippet="The first autonomous AI software engineer"
        )
        assert dedup.window_size == 1

        # 3. Search with a semantically similar Devin title should match cluster-devin
        match_devin = await dedup.find_similar(
            title="Devin: The autonomous developer",
            snippet="Cognition launched Devin AI today",
            threshold=0.8
        )
        assert match_devin is not None
        assert match_devin[0] == "cluster-devin"
        assert pytest.approx(match_devin[1]) == 1.0

        # 4. Search with a different product (Cursor) should NOT match
        match_cursor = await dedup.find_similar(
            title="Cursor code editor",
            snippet="AI pair programming autocomplete",
            threshold=0.8
        )
        assert match_cursor is None
