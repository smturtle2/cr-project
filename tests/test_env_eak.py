from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.util.env_eak import ensure_env_eak, load_env_eak, prepare_b2_environment


class EnvEakTest(unittest.TestCase):
    def test_env_eak_is_encrypted_and_loads_values_without_renaming_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "env"
            eak_path = Path(tmp_dir) / "env.eak"
            env_path.write_text(
                'B2_BUCKET="unit-bucket"\n'
                'B2_ENDPOINT="https://unit.example"\n'
                "B2_KEY_ID='unit-key-id'\n"
                "B2_APP_KEY=unit-app-key\n",
                encoding="utf-8",
            )

            ensure_env_eak(env_path=env_path, eak_path=eak_path)

            payload_text = eak_path.read_text(encoding="utf-8")
            self.assertNotIn("B2_BUCKET", payload_text)
            self.assertNotIn("unit-bucket", payload_text)
            payload = json.loads(payload_text)
            self.assertEqual(payload["cipher"], "fernet")

            with mock.patch.dict(os.environ, {}, clear=True):
                load_env_eak(eak_path=eak_path)

                self.assertEqual(os.environ["B2_BUCKET"], "unit-bucket")
                self.assertEqual(os.environ["B2_ENDPOINT"], "https://unit.example")
                self.assertEqual(os.environ["B2_KEY_ID"], "unit-key-id")
                self.assertEqual(os.environ["B2_APP_KEY"], "unit-app-key")

            self.assertEqual(stat.S_IMODE(eak_path.stat().st_mode), 0o600)

    def test_env_eak_regenerates_when_env_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "env"
            eak_path = Path(tmp_dir) / "env.eak"
            env_path.write_text("B2_BUCKET=first\n", encoding="utf-8")
            ensure_env_eak(env_path=env_path, eak_path=eak_path)
            first_payload = json.loads(eak_path.read_text(encoding="utf-8"))

            env_path.write_text("B2_BUCKET=second\n", encoding="utf-8")
            ensure_env_eak(env_path=env_path, eak_path=eak_path)
            second_payload = json.loads(eak_path.read_text(encoding="utf-8"))

            self.assertNotEqual(first_payload["source_sha256"], second_payload["source_sha256"])
            with mock.patch.dict(os.environ, {}, clear=True):
                load_env_eak(eak_path=eak_path)
                self.assertEqual(os.environ["B2_BUCKET"], "second")

    def test_env_eak_regenerates_when_existing_eak_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "env"
            eak_path = Path(tmp_dir) / "env.eak"
            env_path.write_text("B2_BUCKET=recovered\n", encoding="utf-8")
            ensure_env_eak(env_path=env_path, eak_path=eak_path)

            payload = json.loads(eak_path.read_text(encoding="utf-8"))
            payload["enc"] = "not-valid-fernet-token"
            eak_path.write_text(json.dumps(payload), encoding="utf-8")

            ensure_env_eak(env_path=env_path, eak_path=eak_path)

            with mock.patch.dict(os.environ, {}, clear=True):
                load_env_eak(eak_path=eak_path)
                self.assertEqual(os.environ["B2_BUCKET"], "recovered")

    def test_prepare_b2_environment_uses_eak_without_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "env"
            eak_path = Path(tmp_dir) / "env.eak"
            env_path.write_text("B2_BUCKET=from-eak\n", encoding="utf-8")
            ensure_env_eak(env_path=env_path, eak_path=eak_path)
            env_path.unlink()

            with mock.patch.dict(os.environ, {}, clear=True):
                prepare_b2_environment("B2", env_path=env_path, eak_path=eak_path)
                self.assertEqual(os.environ["B2_BUCKET"], "from-eak")

    def test_prepare_b2_environment_requires_eak_or_env_for_b2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "env"
            eak_path = Path(tmp_dir) / "env.eak"

            with self.assertRaisesRegex(RuntimeError, "env.eak"):
                prepare_b2_environment("B2", env_path=env_path, eak_path=eak_path)

    def test_prepare_b2_environment_skips_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepare_b2_environment(
                "local",
                env_path=Path(tmp_dir) / "env",
                eak_path=Path(tmp_dir) / "env.eak",
            )


if __name__ == "__main__":
    unittest.main()
