"""Pytest conftest: mock GPU-only dependencies so tests run on CPU."""
import sys
from unittest.mock import MagicMock

# flash_attn is GPU-only; mock it so transitive imports don't fail on CPU
for mod_name in ("flash_attn_interface", "flash_attn"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
