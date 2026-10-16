"""Tests for the live-API client script (network calls are mocked)."""

from unittest import mock

import query_live_api


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.ok = status_code < 400

    def json(self):
        return self._body


def test_resolve_base_url_prefers_argument_then_environment(monkeypatch):
    monkeypatch.setenv("LIVE_API_URL", "https://from-env.example.com")
    assert query_live_api.resolve_base_url(
        ["script", "https://from-arg.example.com"]
    ) == "https://from-arg.example.com"
    assert query_live_api.resolve_base_url(
        ["script"]
    ) == "https://from-env.example.com"
    monkeypatch.delenv("LIVE_API_URL")
    assert query_live_api.resolve_base_url(
        ["script"]
    ) == query_live_api.LIVE_API_URL


def test_main_refuses_to_run_with_placeholder_url(capsys, monkeypatch):
    monkeypatch.delenv("LIVE_API_URL", raising=False)
    with mock.patch.object(query_live_api, "LIVE_API_URL",
                           "https://YOUR-APP-NAME.onrender.com"):
        with mock.patch("requests.post") as post:
            exit_code = query_live_api.main(["script"])
    assert exit_code == 2
    post.assert_not_called()
    assert "LIVE_API_URL" in capsys.readouterr().out


def test_main_prints_status_code_and_inference_result(capsys):
    fake = FakeResponse(200, {"prediction": ">50K"})
    with mock.patch("requests.post", return_value=fake) as post:
        exit_code = query_live_api.main(["script", "https://api.example.com/"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "HTTP status code: 200" in out
    assert "{'prediction': '>50K'}" in out
    args, kwargs = post.call_args
    assert args[0] == "https://api.example.com/predict"
    assert kwargs["json"] == query_live_api.SAMPLE_PERSON
