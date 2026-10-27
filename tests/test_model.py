from __future__ import annotations

from src.model import splade


def test_remote_model_load_is_revision_pinned(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSparseEncoder:
        def __init__(self, source, **kwargs):
            captured.update(source=source, **kwargs)
            self.max_seq_length = None

        def eval(self):
            return self

    monkeypatch.setattr(splade, "SparseEncoder", FakeSparseEncoder)
    config = {
        "name_or_path": "owner/model",
        "local_path": str(tmp_path / "missing"),
        "revision": "abc123",
        "local_files_only": False,
        "use_safetensors": False,
        "trust_remote_code": False,
        "max_length": 256,
    }

    model = splade._load_model(config, tmp_path, "cpu")

    assert model.max_seq_length == 256
    assert captured["source"] == "owner/model"
    assert captured["revision"] == "abc123"
    assert captured["model_kwargs"] == {"use_safetensors": False}
