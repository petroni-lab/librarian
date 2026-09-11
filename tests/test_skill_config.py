"""Run with: uv run python tests/test_skill_config.py."""
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/librarian/scripts'))
import _runs
from step1_query_prompt import render_query_prompt

config = replace(_runs.load_runtime_config(), num_subqueries=3, paragraphs_per_subquery=1)
restored = _runs.config_from_manifest({'config': asdict(config)})
agent = _runs.build_agent(restored)
assert agent._num_subqueries == 3
assert agent._paragraphs_per_subquery == 1
prompt = render_query_prompt('test question', restored.query_budget_guidance, restored.num_subqueries)
assert 'AT MOST 3' in prompt and '{max_queries}' not in prompt
print('Plugin config handoff and query prompt passed.')
