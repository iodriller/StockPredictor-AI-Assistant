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


def test_dashboard_binds_to_localhost_by_default(monkeypatch) -> None:
    seen = {}

    def fake_call(command):
        seen["command"] = command
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    assert cli.main(["dashboard", "--config", "configs/default.example.yaml"]) == 0

    command = seen["command"]
    address_index = command.index("--server.address")
    assert command[address_index + 1] == "127.0.0.1"
