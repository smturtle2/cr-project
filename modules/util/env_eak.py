from __future__ import annotations

import binascii
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / "env"
DEFAULT_EAK_PATH = PROJECT_ROOT / "env.eak"

_EAK_VERSION = 1
_EAK_CIPHER = "fernet"


def _source_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_text(payload: bytes, *, path: Path) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} must be UTF-8 encoded") from exc


def _read_eak_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _eak_matches_source(path: Path, source_hash: str) -> bool:
    payload = _read_eak_payload(path)
    if payload is None:
        return False
    if (
        payload.get("version") != _EAK_VERSION
        or payload.get("cipher") != _EAK_CIPHER
        or payload.get("source_sha256") != source_hash
        or not isinstance(payload.get("key"), str)
        or not isinstance(payload.get("enc"), str)
    ):
        return False
    try:
        return _source_sha256(_decrypt_eak_payload(path)) == source_hash
    except RuntimeError:
        return False


def generate_env_eak(
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
    eak_path: str | os.PathLike[str] = DEFAULT_EAK_PATH,
) -> None:
    env_path = Path(env_path)
    eak_path = Path(eak_path)
    source = env_path.read_bytes()
    _decode_text(source, path=env_path)

    key = Fernet.generate_key()
    encrypted = Fernet(key).encrypt(source)
    payload = {
        "version": _EAK_VERSION,
        "cipher": _EAK_CIPHER,
        "key": key.decode("ascii"),
        "enc": encrypted.decode("ascii"),
        "source_sha256": _source_sha256(source),
    }

    eak_path.parent.mkdir(parents=True, exist_ok=True)
    eak_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    try:
        eak_path.chmod(0o600)
    except OSError:
        pass


def ensure_env_eak(
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
    eak_path: str | os.PathLike[str] = DEFAULT_EAK_PATH,
) -> None:
    env_path = Path(env_path)
    eak_path = Path(eak_path)
    if not env_path.exists():
        return

    source = env_path.read_bytes()
    _decode_text(source, path=env_path)
    if _eak_matches_source(eak_path, _source_sha256(source)):
        return
    generate_env_eak(env_path=env_path, eak_path=eak_path)


def _decrypt_eak_payload(path: Path) -> bytes:
    payload = _read_eak_payload(path)
    if payload is None:
        raise RuntimeError(f"{path.name} is missing or invalid")
    if payload.get("version") != _EAK_VERSION or payload.get("cipher") != _EAK_CIPHER:
        raise RuntimeError(f"{path.name} has an unsupported format")

    key = payload.get("key")
    encrypted = payload.get("enc")
    if not isinstance(key, str) or not isinstance(encrypted, str):
        raise RuntimeError(f"{path.name} is missing encrypted environment data")

    try:
        key_bytes = key.encode("ascii")
        encrypted_bytes = encrypted.encode("ascii")
        return Fernet(key_bytes).decrypt(encrypted_bytes)
    except (InvalidToken, ValueError, TypeError, binascii.Error) as exc:
        raise RuntimeError(f"{path.name} could not be decrypted") from exc


def _normalize_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_eak(eak_path: str | os.PathLike[str] = DEFAULT_EAK_PATH) -> None:
    eak_path = Path(eak_path)
    decrypted = _decrypt_eak_payload(eak_path)
    text = _decode_text(decrypted, path=eak_path)

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            raise ValueError(f"{eak_path.name} contains an invalid environment line at {line_number}")
        os.environ[key] = _normalize_env_value(value)


def prepare_b2_environment(
    cache_src: str,
    *,
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
    eak_path: str | os.PathLike[str] = DEFAULT_EAK_PATH,
) -> None:
    if cache_src.strip().lower() != "b2":
        return

    ensure_env_eak(env_path=env_path, eak_path=eak_path)
    eak_path = Path(eak_path)
    if not eak_path.exists():
        raise RuntimeError(f"B2 cache requires {eak_path.name}; create {Path(env_path).name} first")
    load_env_eak(eak_path=eak_path)
