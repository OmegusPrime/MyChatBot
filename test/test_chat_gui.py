"""Exercise desktop event handling with real Tk widgets and a small fake service."""
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import tkinter as tk
    from chat_gui import ChatBotGUI
except ImportError:
    tk = None


class FakeService:
    status = "Ready for testing"

    def __init__(self):
        self.history = []
        self.closed = threading.Event()

    def reply(self, message):
        self.history.append(message)
        return "Answer to " + message

    def reset(self):
        self.history.clear()

    def close(self):
        self.closed.set()


@unittest.skipIf(tk is None, "Tkinter is unavailable")
class TestDesktopConversation(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.root.withdraw()
        self.app = None

    def tearDown(self):
        if self.app is not None and not self.app._closed.is_set():
            self.app._on_close()
        elif self.app is None:
            self.root.destroy()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(predicate(), "The desktop did not finish its background operation")

    def test_send_and_new_chat_update_widgets_and_memory(self):
        service = FakeService()
        with patch("chat_gui.ChatService", return_value=service):
            self.app = ChatBotGUI(self.root, Path("unused"))
            self.until(lambda: not self.app._busy)
            self.app.entry_field.insert(0, "Remember my name")
            self.app._on_send()
            self.until(lambda: not self.app._busy)
            transcript = self.app.chat_display.get("1.0", "end")
            self.assertIn("You: Remember my name", transcript)
            self.assertIn("Chatbot: Answer to Remember my name", transcript)
            self.assertEqual(service.history, ["Remember my name"])
            self.app._on_reset()
            self.until(lambda: not self.app._busy)
            self.assertEqual(service.history, [])
            self.assertNotIn("Remember my name", self.app.chat_display.get("1.0", "end"))
            self.app._on_close()
            self.assertTrue(service.closed.wait(5))

    def test_startup_failure_can_retry_successfully(self):
        service = FakeService()
        with patch("chat_gui.ChatService", side_effect=[RuntimeError("Missing model"), service]):
            self.app = ChatBotGUI(self.root, Path("unused"))
            self.until(lambda: not self.app._busy)
            self.assertIn("Missing model", self.app.chat_display.get("1.0", "end"))
            self.assertEqual(str(self.app.send_button.cget("state")), "disabled")
            self.app._start_loading()
            self.until(lambda: not self.app._busy)
            self.assertEqual(str(self.app.send_button.cget("state")), "normal")

    def test_close_during_model_load_closes_late_created_service(self):
        service = FakeService()
        entered = threading.Event()
        release = threading.Event()

        def load(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test load timeout")
            return service

        with patch("chat_gui.ChatService", side_effect=load):
            self.app = ChatBotGUI(self.root, Path("unused"))
            self.assertTrue(entered.wait(5))
            try:
                self.app._on_close()
            finally:
                release.set()
            self.assertTrue(service.closed.wait(5))

    def test_close_during_generation_releases_waiting_worker(self):
        service = FakeService()
        entered = threading.Event()

        def reply(message):
            entered.set()
            if not service.closed.wait(5):
                raise RuntimeError("test reply timeout")
            raise RuntimeError("closed")

        service.reply = reply
        with patch("chat_gui.ChatService", return_value=service):
            self.app = ChatBotGUI(self.root, Path("unused"))
            self.until(lambda: not self.app._busy)
            self.app.entry_field.insert(0, "Hello")
            self.app._on_send()
            self.assertTrue(entered.wait(5))
            self.app._on_close()
            self.assertTrue(service.closed.wait(5))


if __name__ == "__main__":
    unittest.main()
