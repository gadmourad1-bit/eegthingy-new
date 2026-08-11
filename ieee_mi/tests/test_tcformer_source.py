from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ieee_mi import baselines, tcformer_source
from scripts import vendor_tcformer


def _synthetic_source(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bytes]:
    payloads = {
        "LICENSE": b"synthetic MIT license\n",
        "models/channel_group_attention.py": b"class Attention: pass\n",
        "models/modules.py": b"class Module: pass\n",
        "models/tcformer.py": b"class TCFormerModule: pass\n",
        "utils/weight_initialization.py": b"def init_weight(x): return x\n",
    }
    contracts = {
        relative: {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in payloads.items()
    }
    monkeypatch.setattr(tcformer_source, "TCFORMER_PINNED_FILES", contracts)
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / tcformer_source.TCFORMER_MANIFEST_FILENAME).write_bytes(
        tcformer_source.canonical_manifest_bytes()
    )
    return payloads


def test_official_tcformer_pin_is_exact() -> None:
    assert tcformer_source.TCFORMER_COMMIT == (
        "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"
    )
    assert tcformer_source.TCFORMER_REPOSITORY == (
        "https://github.com/Altaheri/TCFormer"
    )
    assert tcformer_source.TCFORMER_LICENSE == "MIT"
    assert set(tcformer_source.TCFORMER_PINNED_FILES) == {
        "LICENSE",
        "models/channel_group_attention.py",
        "models/modules.py",
        "models/tcformer.py",
        "utils/weight_initialization.py",
    }
    assert all(
        len(contract["sha256"]) == 64
        and type(contract["size_bytes"]) is int
        and contract["size_bytes"] > 0
        for contract in tcformer_source.TCFORMER_PINNED_FILES.values()
    )


def test_vendored_tcformer_source_verifies_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _synthetic_source(tmp_path, monkeypatch)
    identity = tcformer_source.verify_tcformer_source(tmp_path)
    assert identity["source_mode"] == "vendored_runtime_snapshot"
    assert identity["commit"] == tcformer_source.TCFORMER_COMMIT
    assert identity["license"] == "MIT"
    assert identity["source_root"] == str(tmp_path)
    assert set(identity["file_identity"]) == set(
        tcformer_source.TCFORMER_PINNED_FILES
    )


def test_vendored_tcformer_source_rejects_tamper_and_noncanonical_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _synthetic_source(tmp_path, monkeypatch)
    source = tmp_path / "models" / "tcformer.py"
    source.write_bytes(payloads["models/tcformer.py"] + b"# tamper\n")
    with pytest.raises(tcformer_source.TCFormerSourceError, match="differs"):
        tcformer_source.verify_tcformer_source(tmp_path)

    source.write_bytes(payloads["models/tcformer.py"])
    manifest = tcformer_source.canonical_manifest()
    (tmp_path / tcformer_source.TCFORMER_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(tcformer_source.TCFormerSourceError, match="canonical"):
        tcformer_source.verify_tcformer_source(tmp_path)


def test_vendored_tcformer_source_rejects_links_fifo_and_ancestor_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    payloads = _synthetic_source(root, monkeypatch)
    source = root / "models" / "tcformer.py"
    hardlink = tmp_path / "hardlink.py"
    os.link(source, hardlink)
    with pytest.raises(tcformer_source.TCFormerSourceError, match="single-link"):
        tcformer_source.verify_tcformer_source(root)

    hardlink.unlink()
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(tcformer_source.TCFormerSourceError, match="single-link"):
        tcformer_source.verify_tcformer_source(root)

    source.unlink()
    source.write_bytes(payloads["models/tcformer.py"])
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(tcformer_source.TCFormerSourceError, match="ancestor"):
        tcformer_source.verify_tcformer_source(alias)


def test_verified_module_executes_only_the_descriptor_read_pinned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "verified.py"
    verified = b"VALUE = 'verified-bytes'\n"
    source.write_bytes(verified)
    original = baselines._read_unique_regular_bytes

    def replace_after_read(path: Path, *, root: Path) -> bytes:
        payload = original(path, root=root)
        path.write_bytes(b"raise RuntimeError('pathname payload executed')\n")
        return payload

    monkeypatch.setattr(
        baselines,
        "_read_unique_regular_bytes",
        replace_after_read,
    )
    module = baselines._load_verified_module(
        "_synthetic_verified_tcformer",
        source,
        root=tmp_path,
        expected_sha256=hashlib.sha256(verified).hexdigest(),
    )
    assert module.VALUE == "verified-bytes"


def test_vendor_script_creates_verified_snapshot_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "checkout"
    source.mkdir()
    payloads = _synthetic_source(source, monkeypatch)
    (source / tcformer_source.TCFORMER_MANIFEST_FILENAME).unlink()
    contracts = dict(tcformer_source.TCFORMER_PINNED_FILES)
    monkeypatch.setattr(vendor_tcformer, "TCFORMER_PINNED_FILES", contracts)

    def fake_git_output(checkout: Path, arguments: tuple[str, ...]) -> str:
        assert checkout == source
        if arguments == ("rev-parse", "HEAD"):
            return tcformer_source.TCFORMER_COMMIT
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if arguments == ("remote", "get-url", "origin"):
            return f"{tcformer_source.TCFORMER_REPOSITORY}.git"
        raise AssertionError(arguments)

    monkeypatch.setattr(vendor_tcformer, "_git_output", fake_git_output)
    destination = tmp_path / "release" / "third_party" / "TCFormer"
    destination.parent.mkdir(parents=True)
    identity = vendor_tcformer.vendor_tcformer(source, destination)
    assert identity["source_mode"] == "vendored_runtime_snapshot"
    assert {
        relative: (destination / relative).read_bytes()
        for relative in payloads
    } == payloads
    with pytest.raises(vendor_tcformer.VendorError, match="refusing to overwrite"):
        vendor_tcformer.vendor_tcformer(source, destination)
