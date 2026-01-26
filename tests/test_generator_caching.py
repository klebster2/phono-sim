import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


@pytest.mark.skip()
def test_caching_performance():
    language = "fra"
    py_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    cache_dir = Path(os.path.expanduser("~/.cache/phono-sim"))
    cache_file = cache_dir / f"{language}_{py_version}.pkl"

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any previous cache file
    if cache_file.exists():
        cache_file.unlink()

    try:
        # First run (should create cache)
        start_time_first = time.time()
        CharsiuGraphemeToPhonemeGenerator(language, use_cache=True)
        end_time_first = time.time()
        first_run_duration = end_time_first - start_time_first
        assert cache_file.exists()

        # Second run (should use cache)
        start_time_second = time.time()
        CharsiuGraphemeToPhonemeGenerator(language, use_cache=True)
        end_time_second = time.time()
        second_run_duration = end_time_second - start_time_second

        # Assert the second run is much faster
        assert second_run_duration < first_run_duration, "Second run should be faster"

    finally:
        # Clean up the cache directory
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
