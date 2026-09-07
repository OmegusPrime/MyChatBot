"""Terminal entry points share a service and always release it on exit."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import chat_cli
import main_and_eval


class TestChatCLI(unittest.TestCase):
    def test_reply_failure_reset_and_exit_use_one_service(self):
        service = Mock(status="Ready")
        service.reply.side_effect = [RuntimeError("Temporary error"), "Hello"]
        output = io.StringIO()
        with (
            patch("chat_cli.ChatService", return_value=service) as factory,
            patch("builtins.input", side_effect=["", "Hi", "Hi again", "/reset", "/exit"]),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(chat_cli.run_terminal_chat(), 0)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(service.reply.call_count, 2)
        service.reset.assert_called_once_with()
        service.close.assert_called_once_with()
        self.assertIn("Temporary error", output.getvalue())
        self.assertIn("Chatbot: Hello", output.getvalue())

    def test_eof_and_interrupt_close_service(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error):
                service = Mock(status="Ready")
                with (
                    patch("chat_cli.ChatService", return_value=service),
                    patch("builtins.input", side_effect=error),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(chat_cli.run_terminal_chat(), 0)
                service.close.assert_called_once_with()

    def test_startup_failure_returns_unsuccessful_exit_code(self):
        with (
            patch("chat_cli.ChatService", side_effect=RuntimeError("Missing model")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(chat_cli.run_terminal_chat(), 1)

    def test_existing_chat_command_delegates_backend_and_exit_status(self):
        with patch("chat_cli.run_terminal_chat", return_value=1) as run:
            self.assertEqual(main_and_eval.main(["--chat", "--backend", "legacy"]), 1)
        self.assertEqual(run.call_args.args[2], "legacy")


if __name__ == "__main__":
    unittest.main()
