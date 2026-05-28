from __future__ import annotations

from stockpredictor import cli


def test_cli_accepts_config_on_subcommand_only(monkeypatch, capsys) -> None:
    seen = {}

    def fake_load_settings(path):
        seen["path"] = path
        return object()

    monkeypatch.setattr(cli, "load_settings", fake_load_settings)
    monkeypatch.setattr(cli, "scan_symbols", lambda *args, **kwargs: [])

    assert cli.main(["scan", "--config", "configs/custom.yaml"]) == 0

    assert seen["path"] == "configs/custom.yaml"
    assert capsys.readouterr().out.strip() == "[]"
