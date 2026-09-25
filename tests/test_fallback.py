import socket
import urllib.request

from app.tools.fallback import mock_web_search


def test_mock_search_is_labeled_deterministic_and_offline(mocker):
    connect = mocker.patch.object(socket, "create_connection", side_effect=AssertionError("network call"))
    urlopen = mocker.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network call"))

    first = mock_web_search("capital of France")
    second = mock_web_search("capital of France")

    assert first == second
    assert first["source"] == "mock_web_search"
    assert "simulated fallback" in first["note"]
    assert "not verified knowledge-base evidence" in first["note"]
    connect.assert_not_called()
    urlopen.assert_not_called()
