import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'wumpus_chase' / 'src'))
from tools.notebook_runtime import load_runtime

@pytest.fixture
def pilot(tmp_path):
    module = load_runtime(tmp_path)
    yield module
    module.WORLD.expected_payoff.cache_clear()
    sys.modules.pop(module.__name__, None)

@pytest.fixture
def tag_cls():
    name = 'base_chase_environment'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'chase/src/env/tag_env.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.TagEnv
