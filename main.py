from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
BACKEND_MAIN_PATH = os.path.join(BACKEND_DIR, "main.py")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if not os.path.exists(BACKEND_MAIN_PATH):
    raise RuntimeError(f"Backend entrypoint not found at {BACKEND_MAIN_PATH}.")

backend_package = sys.modules.get("backend")
if backend_package is None:
    backend_package = types.ModuleType("backend")
    backend_package.__path__ = [BACKEND_DIR]
    sys.modules["backend"] = backend_package

spec = importlib.util.spec_from_file_location("backend.main", BACKEND_MAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load backend module from {BACKEND_MAIN_PATH}.")

backend_main = importlib.util.module_from_spec(spec)
sys.modules["backend.main"] = backend_main
spec.loader.exec_module(backend_main)

app = backend_main.app
