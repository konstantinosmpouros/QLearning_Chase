"""Load the real notebook definitions without launching experiments or analysis.

Only cells explicitly tagged qcatch-definitions are executed. Test overrides are
inserted at the configuration boundary, before defaults bind to CFG. Algorithm
bodies are executed verbatim; outputs/plots and the built-in check invocation are
not executed. This avoids maintaining a second implementation for tests.
"""
import ast
import contextlib
import io
import json
import sys
import types
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'Chase_Population_Robustness_Tested.ipynb'


def load_runtime(output_dir, notebook=NOTEBOOK, name='qcatch_pilot_test'):
    nb = json.loads(Path(notebook).read_text(encoding='utf-8'))
    module = types.ModuleType(name)
    module.__file__ = str(notebook)
    sys.modules[name] = module  # dataclasses and checkpoint pickle need identity
    found = set()
    for cell in nb['cells']:
        tags = cell.get('metadata', {}).get('tags', [])
        if 'qcatch-definitions' not in tags:
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        if 'qcatch-config' in tags:
            found.add('config')
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {'PROFILE', 'RUN_EXPERIMENTS', 'OUTPUT_ROOT'}:
                            value = {'PROFILE': repr('smoke'), 'RUN_EXPERIMENTS': 'False',
                                     'OUTPUT_ROOT': 'Path('+repr(str(output_dir))+')'}[target.id]
                            node.value = ast.parse(value, mode='eval').body
            ast.fix_missing_locations(tree)
        if 'qcatch-functions-only' in tags:
            tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(notebook), 'exec'), module.__dict__)
    if found != {'config'} or not hasattr(module, 'evaluate_condition'):
        raise ValueError('Missing tagged notebook definitions')
    return module
