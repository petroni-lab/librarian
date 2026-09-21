"""Direct-session smoke checks. Run with: uv run python tests/test_antigravity.py."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/librarian/scripts'))
import _direct_session as sessions

with patch.dict('os.environ', {}, clear=True):
    command = sessions._codex_command(
        '/bin/codex', Path('/tmp/prompt.md'), Path('/tmp/schema.json'), Path('/tmp/output.json')
    )
    assert command[command.index('-m') + 1] == 'gpt-5.6-luna'
    assert command[command.index('-c') + 1] == 'model_reasoning_effort="low"'

with patch.dict('os.environ', {
    'LIBRARIAN_CODEX_MODEL': '',
    'LIBRARIAN_CODEX_EFFORT': '',
}, clear=True):
    command = sessions._codex_command(
        '/bin/codex', Path('/tmp/prompt.md'), Path('/tmp/schema.json'), Path('/tmp/output.json')
    )
    assert '-m' not in command
    assert '-c' not in command

with tempfile.TemporaryDirectory() as directory:
    prompt = Path(directory) / 'prompt.md'
    output = Path(directory) / 'output.json'
    prompt.write_text('Question with "quotes"\nand Unicode: α', encoding='utf-8')

    for key in ('queries', 'relevant_ids'):
        def fake_run(command, **kwargs):
            assert command[0] == '/bin/agy'
            assert command[1:5] == ['--input-format', 'stream-json', '--output-format', 'stream-json']
            assert json.loads(command[6])['required'] == [key]
            event = json.load(kwargs['stdin'])
            assert event == {'event': 'user', 'message': {'content': prompt.read_text()}}
            assert kwargs['timeout'] == 180
            response = json.dumps({key: ['example']})
            stream = json.dumps({'event': 'init'}) + '\n' + json.dumps({
                'event': 'result', 'result': {'status': 'SUCCESS', 'response': response},
            })
            return subprocess.CompletedProcess(command, 0, stream, '')

        with patch.object(sessions.shutil, 'which', return_value='/bin/agy') as which, \
             patch.object(sessions.subprocess, 'run', side_effect=fake_run):
            sessions.run_direct_session('antigravity', prompt, output, key)
            which.assert_called_once_with('agy')
        assert json.loads(output.read_text()) == {key: ['example']}

    for raw in ('', 'not json', '[]', '{"event":"result","result":{"status":"ERROR"}}',
                '{"event":"result","result":{"status":"SUCCESS","response":null}}'):
        try:
            sessions._antigravity_response(raw)
        except SystemExit:
            pass
        else:
            raise AssertionError(f'Accepted invalid stream: {raw}')

    with patch.object(sessions.shutil, 'which', return_value=None):
        try:
            sessions.run_direct_session('antigravity', prompt, output, 'queries')
        except SystemExit as error:
            assert '(agy)' in str(error)
        else:
            raise AssertionError('Missing CLI accepted')

print('Codex defaults and Antigravity session checks passed.')
