import os

import pytest

from backend.demo import main


@pytest.mark.integration
def test_demo_runs() -> None:
    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("requires a running Ollama server and local models")

    main()
