"""Content checks shared by the public-tree gate and the runtime boundary guards."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

# Copied from manyruns v0.1.0 under MIT; upstream notice is retained in LICENSE.
# Lab-private name hashes may be supplied locally; no private name list ships.
FORBIDDEN_NAME_HASHES = frozenset()


def forbidden_count(text: str, hashes=FORBIDDEN_NAME_HASHES) -> int:
    """Hash words and identifier components; return counts, never matched text."""
    # Keep whole identifiers as well as snake/kebab/camel components and dotted imports.
    words = re.findall(r"[A-Za-z0-9]+", text)
    tokens = set(words)
    for word in words:
        tokens.update(re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", word))
    return sum(hashlib.sha256(token.lower().encode()).hexdigest() in hashes for token in tokens)


_CREDENTIALS = re.compile(
    rb"(?:\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    rb"|\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"
    rb"|\bsk-(?:ant-api\d+-)?[A-Za-z0-9_-]{32,}\b"
    rb"|\bxox[baprs]-[A-Za-z0-9-]{20,}\b|\bAIza[A-Za-z0-9_-]{35}\b"
    rb"|-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    rb"|https?://[^\s/:@]+:[^\s/@]+@"
    rb"|(?i:(?:api[_-]?key|access[_-]?token|secret|password)[\"']?\s*[:=]\s*[\"']"
    rb"[A-Za-z0-9_+/=-]{32,}[\"']))"
)
_REGISTRIES = re.compile(
    rb"pkg\.dev\b"
    rb"|https?://[^\s/\"'<>]*(?:\.pkg\.github\.com|\.jfrog\.io|\.codeartifact\."
    rb"|\.cloudsmith\.io)[^\s\"'<>]*"
    rb"|https?://[^\s\"'<>]*/(?:artifactory|repository)/[^\s\"'<>]*"
    rb"|https?://[^\s/\"'<>]*(?:private|internal)[^\s/\"'<>]*/[^\s\"'<>]*"
    rb"(?:simple|pypi|registry|index)[^\s\"'<>]*",
    re.IGNORECASE,
)
_MACHINE_PATHS = re.compile(
    rb"/(?:Users|home)/[^/\s\x00\"'<>]+"
    rb"|/(?:private/)?tmp/(?:claude|codex)-[^/\s\x00\"'<>]+"
    rb"|/(?:private/)?var/folders/[A-Za-z0-9_]{2}/[A-Za-z0-9_-]{20,}"
    rb"|/pytest-of-[\w.-]+"
)


def violations(path: str, content: bytes, hashes=FORBIDDEN_NAME_HASHES) -> Counter:
    """Only category counts escape this function, even when a filename is sensitive."""
    result = Counter()
    result['forbidden name'] = forbidden_count(path, hashes) + forbidden_count(
        content.decode('utf-8', errors='replace'), hashes)
    parts = Path(path).parts
    result['environment file'] = int(any(part == '.env' or part.startswith('.env.') for part in parts))
    result['credential'] = len(_CREDENTIALS.findall(content))
    result['private registry'] = len(_REGISTRIES.findall(content))
    result['machine path'] = (len(_MACHINE_PATHS.findall(path.encode('utf-8')))
                              + len(_MACHINE_PATHS.findall(content)))
    if Path(path).suffix.lower() in {'.h5ad', '.h5', '.loom', '.mtx', '.npy', '.npz',
                                      '.csv', '.tsv', '.parquet', '.arrow'}:
        result['data file'] = 1
    return +result
