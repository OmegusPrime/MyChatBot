"""Managed model transport and lifecycle checks using no network or model files."""

import json
import subprocess
import threading
import unittest
from unittest.mock import Mock, call, patch

from local_model import LocalModelProvider, _ServerHTTPError


class TestLocalModelProvider(unittest.TestCase):
    def make_provider(self):
        provider = object.__new__(LocalModelProvider)
        provider.context_length = 4096
        provider.request_timeout = 10
        provider._closed = threading.Event()
        provider._process_lock = threading.Lock()
        provider._process = Mock()
        provider._process.poll.return_value = None
        provider._stderr_thread = None
        provider._api_key = "test-session-key"
        provider._exact_token_count = True
        provider.port = 12345
        return provider

    def connection(self, raw=b'{"status":"ok"}', status=200):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = status
        response.read.return_value = raw
        return connection

    def test_request_uses_authenticated_loopback_and_closes_connection(self):
        provider = self.make_provider()
        connection = self.connection()
        payload = {"messages": [{"role": "user", "content": "Hello, Maya!"}]}
        with patch("local_model.http.client.HTTPConnection", return_value=connection) as factory:
            result = provider._request("POST", "/test", payload)
        self.assertEqual(result, {"status": "ok"})
        factory.assert_called_once_with("127.0.0.1", 12345, timeout=10)
        request = connection.request.call_args
        self.assertEqual(request.args, ("POST", "/test"))
        self.assertEqual(json.loads(request.kwargs["body"]), payload)
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer test-session-key")
        connection.close.assert_called_once_with()

    def test_invalid_empty_or_oversized_http_bodies_raise_and_close(self):
        provider = self.make_provider()
        for raw in (b"", b"not-json", b"[]", b"null", b"\xff", b'{"error":"bad"}', b"x" * (4 * 1024 * 1024 + 1)):
            with self.subTest(raw=raw[:30]):
                connection = self.connection(raw)
                with patch("local_model.http.client.HTTPConnection", return_value=connection):
                    with self.assertRaises(RuntimeError):
                        provider._request("GET", "/health")
                connection.close.assert_called_once_with()

    def test_http_errors_and_redirects_are_reported_without_following(self):
        provider = self.make_provider()
        for status in (301, 302, 401, 404, 500):
            with self.subTest(status=status):
                connection = self.connection(status=status)
                with patch("local_model.http.client.HTTPConnection", return_value=connection) as factory:
                    with self.assertRaises(_ServerHTTPError) as caught:
                        provider._request("POST", "/v1/chat/completions", {"messages": []})
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(connection.request.call_count, 1)
                connection.close.assert_called_once_with()

    def test_timeout_and_disconnect_are_actionable_and_close_transport(self):
        provider = self.make_provider()
        for failure, message in ((TimeoutError("late"), "timed out"), (OSError("gone"), "Cannot reach")):
            with self.subTest(failure=failure):
                connection = self.connection()
                connection.getresponse.side_effect = failure
                with patch("local_model.http.client.HTTPConnection", return_value=connection):
                    with self.assertRaisesRegex(RuntimeError, message):
                        provider._request("POST", "/v1/chat/completions", {})
                connection.close.assert_called_once_with()

    def test_closed_or_dead_process_is_detected_before_opening_transport(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                provider = self.make_provider()
                if closed:
                    provider._closed.set()
                else:
                    provider._process.poll.return_value = 1
                with patch("local_model.http.client.HTTPConnection") as factory:
                    with self.assertRaises(RuntimeError):
                        provider._request("GET", "/health")
                factory.assert_not_called()

    def test_token_count_uses_complete_chat_template_and_special_tokens(self):
        provider = self.make_provider()
        messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}]
        provider._request = Mock(side_effect=[{"prompt": "the complete chat template"}, {"tokens": [1, 2, 3, 4]}])
        self.assertEqual(provider.count_tokens(messages), 4)
        self.assertEqual(provider._request.call_args_list, [
            call("POST", "/apply-template", {"messages": messages}),
            call("POST", "/tokenize", {"content": "the complete chat template", "add_special": True, "parse_special": True}),
        ])

    def test_missing_template_endpoint_uses_conservative_cached_fallback(self):
        messages = [{"role": "user", "content": "Hello, 世界"}]
        for status in (404, 501):
            with self.subTest(status=status):
                provider = self.make_provider()
                provider._request = Mock(side_effect=_ServerHTTPError(status, "Unavailable"))
                first = provider.count_tokens(messages)
                second = provider.count_tokens(messages)
                self.assertEqual(first, second)
                self.assertGreaterEqual(first, len(messages[0]["content"].encode("utf-8")))
                self.assertEqual(provider._request.call_count, 1)

    def test_other_template_errors_are_not_hidden_by_estimation(self):
        provider = self.make_provider()
        provider._request = Mock(side_effect=_ServerHTTPError(401, "Unauthorized"))
        with self.assertRaises(_ServerHTTPError):
            provider.count_tokens([{"role": "user", "content": "Hello"}])
        self.assertTrue(provider._exact_token_count)

    def test_malformed_template_and_token_results_are_rejected(self):
        for responses in ([{}], [{"prompt": ""}], [{"prompt": "valid"}, {"tokens": []}], [{"prompt": "valid"}, {"tokens": "invalid"}]):
            with self.subTest(responses=responses):
                provider = self.make_provider()
                provider._request = Mock(side_effect=responses)
                with self.assertRaises(RuntimeError):
                    provider.count_tokens([{"role": "user", "content": "Hello"}])

    def test_completion_rejects_malformed_or_empty_choices(self):
        invalid = [
            {}, {"choices": []}, {"choices": [None]}, {"choices": [{}]},
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": "  \n"}}]},
        ]
        for result in invalid:
            with self.subTest(result=result):
                provider = self.make_provider()
                provider._request = Mock(return_value=result)
                with self.assertRaises(RuntimeError):
                    provider.complete([{"role": "user", "content": "Hi"}], max_new_tokens=64, temperature=0.3)

    def test_completion_preserves_request_and_explains_length_cutoff(self):
        provider = self.make_provider()
        provider._request = Mock(return_value={"choices": [{"message": {"content": "  Part of the answer.  "}, "finish_reason": "length"}]})
        messages = [{"role": "user", "content": "Explain photosynthesis."}]
        result = provider.complete(messages, max_new_tokens=64, temperature=0.3)
        self.assertTrue(result.startswith("Part of the answer."))
        self.assertIn("length limit", result)
        method, endpoint, payload = provider._request.call_args.args
        self.assertEqual((method, endpoint), ("POST", "/v1/chat/completions"))
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertFalse(payload["stream"])

    def test_close_terminates_owned_process_once_and_closes_output(self):
        provider = self.make_provider()
        process = provider._process
        provider.close()
        provider.close()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)
        process.kill.assert_not_called()
        process.stderr.close.assert_called_once_with()

    def test_close_kills_process_if_graceful_stop_times_out(self):
        provider = self.make_provider()
        process = provider._process
        process.wait.side_effect = [subprocess.TimeoutExpired("test-server", 3), 0]
        provider.close()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)
        process.stderr.close.assert_called_once_with()

    def test_startup_timeout_stops_the_process_it_started(self):
        process = Mock()
        process.poll.return_value = None
        with (
            patch("local_model.Path.is_file", return_value=True),
            patch("local_model.socket.socket") as socket_factory,
            patch("local_model.subprocess.Popen", return_value=process),
            patch("local_model.threading.Thread"),
            patch("local_model.time.monotonic", side_effect=[0, 2]),
        ):
            socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            with self.assertRaisesRegex(RuntimeError, "too long to load"):
                LocalModelProvider(".", startup_timeout=1)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)
        process.stderr.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
