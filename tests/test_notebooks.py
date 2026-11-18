"""Run every notebook from top to bottom. A notebook that raises an error fails the test."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
