"""One gate over every tracked file, including prose, lockfiles and binary payloads."""
from collections import Counter
import hashlib
from pathlib import Path
import subprocess

import pytest

from public_safety import violations

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.source_checkout
def test_tracked_tree_is_public_safe():
    __tracebackhide__ = True
    assert ROOT.is_dir(), 'source root moved'
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).split(b'\0')
    assert any(files), 'the public-safety gate scanned no tracked files'
    counts = Counter()
    for name in files:
        if name:
            path = name.decode('utf-8')
            counts.update(violations(path, (ROOT / path).read_bytes()))
    if counts:
        pytest.fail(f'Public-safety violations (category counts only): {dict(counts)}',
                    pytrace=False)


def test_gate_checks_names_in_paths_and_binary_contents():
    token = 'Sentinel'
    hashes = {hashlib.sha256(token.lower().encode()).hexdigest()}
    assert violations(f'notes/{token}.md', b'', hashes)['forbidden name']
    assert violations('asset.bin', b'\xff' + token.encode() + b'\x00', hashes)['forbidden name']


@pytest.mark.parametrize('kind', ['cloud', 'github', 'model', 'slack', 'private-key',
                                  'url-password', 'api-key'])
def test_credential_shapes_are_rejected(kind):
    samples = {
        'cloud': 'AK' + 'IA' + 'A' * 16,
        'github': 'gh' + 'p_' + 'a' * 36,
        'model': 'sk' + '-ant-api03-' + 'a' * 80,
        'slack': 'xo' + 'xb-' + '1' * 30,
        'private-key': '-----BEGIN ' + 'PRIVATE KEY-----',
        'url-password': 'https://' + 'user:password@host/simple',
        'api-key': 'api_key = "' + 'a' * 40 + '"',
    }
    assert violations('example.txt', samples[kind].encode())['credential']


@pytest.mark.parametrize('host_path', [
    'region-python.' + 'pkg' + '.dev/project/repo/simple',
    'packages.' + 'jfrog' + '.io/artifactory/pypi',
    'domain.' + 'codeartifact' + '.region.amazonaws.com/pypi/repo/simple',
    'pypi.internal.example/simple',
    'host.example/artifactory/api/pypi/simple',
])
def test_private_registry_shapes_are_rejected(host_path):
    assert violations('config.toml', ('https://' + host_path).encode())['private registry']


def test_public_index_is_allowed():
    assert not violations('config.toml', b'https://pypi.org/simple')


@pytest.mark.parametrize('root', ['Users', 'home'])
@pytest.mark.parametrize('suffix', ['/', '/project/requirements.in', ''])
def test_absolute_user_home_paths_are_rejected(root, suffix):
    machine_path = f'/{root}/sentinel-user{suffix}'
    assert violations('notes.md', machine_path.encode())['machine path']
    assert violations(machine_path, b'')['machine path']


@pytest.mark.parametrize('root', ['/private/tmp', '/tmp'])
@pytest.mark.parametrize('session', ['claude', 'codex'])
def test_temporary_session_paths_are_rejected(root, session):
    machine_path = f'{root}/{session}-123/session/requirements.in'
    assert violations('tool.lock', machine_path.encode())['machine path']
    assert violations('asset.bin', b'\xff' + machine_path.encode() + b'\x00')['machine path']


@pytest.mark.parametrize('root', ['/var/folders', '/private/var/folders'])
def test_macos_per_user_temporary_paths_are_rejected(root):
    machine_path = f'{root}/xx/{"a" * 30}/T/plots/phate.png'
    assert violations('notes.md', machine_path.encode())['machine path']


def test_pytest_user_temporary_paths_are_rejected():
    machine_path = '/tmp/pytest-of-' + 'sentinel-user/pytest-1/plots/phate.png'
    assert violations('notes.md', machine_path.encode())['machine path']


@pytest.mark.parametrize('content', [
    b'# Run with cwd=/private/tmp to isolate tool output.',
    b'# cwd=/private/tmp/',
    b'/tmp/example/output.json',
    b'~/project/requirements.in',
    b'$HOME/project/requirements.in',
])
def test_generic_portable_paths_are_allowed(content):
    assert not violations('notes.md', content)


def test_environment_files_are_rejected():
    assert violations('.env', b'')['environment file']
    assert violations('nested/.env', b'')['environment file']
    assert violations('nested/.env.local', b'')['environment file']


@pytest.mark.parametrize('suffix', ['h5ad', 'h5', 'loom', 'mtx', 'npy', 'npz',
                                         'csv', 'tsv', 'parquet', 'arrow'])
def test_data_files_have_no_placeholder_exception(suffix):
    assert violations(f'fixture.{suffix}', b'')['data file']


def test_gate_rejects_a_tracked_reintroduction_without_printing_it(tmp_path, monkeypatch):
    """Exercise discovery and the failure message, not just the token matcher."""
    import functools
    import sys

    token = 'Sentinel'
    hashes = {hashlib.sha256(token.lower().encode()).hexdigest()}
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / f'{token}.md').write_text(f'A paragraph about {token.upper()}.')
    (tmp_path / 'uv.lock').write_text(f'name = "{token}"\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    monkeypatch.setattr(sys.modules[__name__], 'violations',
                        functools.partial(violations, hashes=hashes))

    with pytest.raises(pytest.fail.Exception) as failure:
        test_tracked_tree_is_public_safe()

    message = str(failure.value)
    assert 'forbidden name' in message
    assert token.lower() not in message.lower()


@pytest.mark.parametrize('root, directory', [
    ('/Users', 'sentinel-user'),
    ('/home', 'sentinel-user'),
    ('/private/tmp', 'claude-' + '123'),
    ('/private/var/folders/xx', 'a' * 30),
    ('/tmp', 'pytest-of-' + 'sentinel-user'),
])
def test_gate_reports_machine_path_counts_without_identifying_text(tmp_path, monkeypatch,
                                                                 root, directory):
    import sys

    machine_path = f'{root}/{directory}/session/requirements.in'
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / 'provenance.lock').write_text(f'# via -r {machine_path}\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)

    with pytest.raises(pytest.fail.Exception) as failure:
        test_tracked_tree_is_public_safe()

    assert str(failure.value) == (
        "Public-safety violations (category counts only): {'machine path': 1}")
