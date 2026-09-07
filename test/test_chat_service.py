"""Conversation regressions without downloading or loading a model."""

import copy
import threading
import unittest

from chat_backend import ChatService


class RecordingProvider:
    def __init__(self, responses=None, *, context_length=1024, tokens_per_message=20):
        self.context_length = context_length
        self.tokens_per_message = tokens_per_message
        self.responses = list(responses or ["Hello! How can I help?"])
        self.calls = []
        self.close_calls = 0

    def count_tokens(self, messages):
        return len(messages) * self.tokens_per_message

    def complete(self, messages, *, max_new_tokens, temperature):
        self.calls.append((copy.deepcopy(messages), max_new_tokens, temperature))
        response = self.responses.pop(0) if self.responses else "A helpful answer."
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.close_calls += 1


class TestChatService(unittest.TestCase):
    def make_service(self, responses=None, **kwargs):
        provider = RecordingProvider(responses, **kwargs)
        service = ChatService(provider=provider)
        self.addCleanup(service.close)
        return service, provider

    def test_reply_preserves_roles_and_follow_up_context(self):
        service, provider = self.make_service(["Nice to meet you, Maya.", "Your name is Maya."])
        self.assertEqual(service.reply("My name is Maya.", max_new_tokens=64), "Nice to meet you, Maya.")
        self.assertEqual(service.reply("What is my name?", max_new_tokens=64), "Your name is Maya.")
        first_prompt = provider.calls[0][0]
        follow_up = provider.calls[1][0]
        self.assertEqual(first_prompt[0]["role"], "system")
        self.assertTrue(first_prompt[0]["content"].strip())
        self.assertEqual(follow_up, [
            first_prompt[0],
            {"role": "user", "content": "My name is Maya."},
            {"role": "assistant", "content": "Nice to meet you, Maya."},
            {"role": "user", "content": "What is my name?"},
        ])
        self.assertEqual([message["role"] for message in service.history],
                         ["user", "assistant", "user", "assistant"])

    def test_greeting_and_compound_question_are_passed_to_model(self):
        service, provider = self.make_service(["Hi!", "Hello! The capital of France is Paris."])
        self.assertEqual(service.reply("Hi", max_new_tokens=64), "Hi!")
        question = "Hi, what is the capital of France?"
        self.assertEqual(service.reply(question, max_new_tokens=64),
                         "Hello! The capital of France is Paris.")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[-1][0][-1], {"role": "user", "content": question})

    def test_reset_clears_context_but_keeps_provider_usable(self):
        service, provider = self.make_service(["I will remember that.", "Hello again."])
        service.reply("My name is Maya.", max_new_tokens=64)
        service.reset()
        self.assertEqual(service.history, [])
        service.reply("Hello again", max_new_tokens=64)
        self.assertEqual([item["role"] for item in provider.calls[-1][0]], ["system", "user"])
        self.assertNotIn("Maya", str(provider.calls[-1][0]))
        self.assertEqual(provider.close_calls, 0)

    def test_failed_request_does_not_poison_follow_up_context(self):
        service, provider = self.make_service([
            "Nice to meet you, Maya.", RuntimeError("The server stopped."), "Your name is Maya.",
        ])
        service.reply("My name is Maya.", max_new_tokens=64)
        original_history = service.history
        with self.assertRaisesRegex(RuntimeError, "server stopped"):
            service.reply("This request fails.", max_new_tokens=64)
        self.assertEqual(service.history, original_history)
        service.reply("What is my name?", max_new_tokens=64)
        self.assertNotIn("This request fails.", str(provider.calls[-1][0]))
        self.assertEqual(provider.calls[-1][0][1]["content"], "My name is Maya.")

    def test_empty_model_answer_is_an_error_and_does_not_enter_history(self):
        service, _provider = self.make_service(["A previous answer.", "   \n\t"])
        service.reply("A previous question.", max_new_tokens=64)
        original_history = service.history
        with self.assertRaises(RuntimeError):
            service.reply("The next question.", max_new_tokens=64)
        self.assertEqual(service.history, original_history)

    def test_context_trimming_removes_only_complete_old_exchanges(self):
        service, provider = self.make_service(
            ["First answer.", "Second answer.", "Third answer."],
            context_length=512, tokens_per_message=100,
        )
        service.reply("First question.", max_new_tokens=64)
        service.reply("Second question.", max_new_tokens=64)
        service.reply("Third question.", max_new_tokens=64)
        prompt = provider.calls[-1][0]
        self.assertEqual([message["role"] for message in prompt],
                         ["system", "user", "assistant", "user"])
        self.assertEqual([message["content"] for message in prompt[1:]],
                         ["Second question.", "Second answer.", "Third question."])
        self.assertEqual(len(service.history) % 2, 0)
        self.assertNotIn("First question.", str(service.history))
        self.assertLessEqual(provider.count_tokens(prompt) + 64 + 16, provider.context_length)

    def test_failed_completion_does_not_commit_context_trimming(self):
        service, _provider = self.make_service(
            ["First answer.", "Second answer.", RuntimeError("Connection lost")],
            context_length=512, tokens_per_message=100,
        )
        service.reply("First question.", max_new_tokens=64)
        service.reply("Second question.", max_new_tokens=64)
        original_history = service.history
        with self.assertRaises(RuntimeError):
            service.reply("Third question.", max_new_tokens=64)
        self.assertEqual(service.history, original_history)

    def test_empty_and_nontext_inputs_are_rejected_before_model_call(self):
        service, provider = self.make_service()
        for value in ("", "  \n\t", None, 42, ["hi"]):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                service.reply(value, max_new_tokens=64)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.history, [])

    def test_oversized_input_is_rejected_instead_of_silently_cut_off(self):
        service, provider = self.make_service(context_length=512)
        with self.assertRaises(ValueError):
            service.reply("A" * (provider.context_length * 8 + 1), max_new_tokens=64)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.history, [])

    def test_prompt_that_cannot_fit_raises_before_completion(self):
        service, provider = self.make_service(context_length=512, tokens_per_message=240)
        with self.assertRaises(ValueError):
            service.reply("A question that cannot fit this context.", max_new_tokens=64)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.history, [])

    def test_invalid_generation_settings_do_not_reach_provider(self):
        service, provider = self.make_service(context_length=512)
        for kwargs in (
            {"max_new_tokens": 0}, {"max_new_tokens": -1}, {"max_new_tokens": 385},
            {"max_new_tokens": 1.5}, {"temperature": -0.1}, {"temperature": 2.1},
            {"temperature": float("nan")}, {"temperature": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                service.reply("Hello", **kwargs)
        self.assertEqual(provider.calls, [])

    def test_history_is_a_defensive_copy(self):
        service, provider = self.make_service()
        service.reply("Remember this.", max_new_tokens=64)
        external_history = service.history
        external_history[0]["content"] = "Altered externally."
        external_history.clear()
        service.reply("What did I say?", max_new_tokens=64)
        self.assertEqual(provider.calls[-1][0][1]["content"], "Remember this.")

    def test_close_is_idempotent_and_closed_service_rejects_work(self):
        service, provider = self.make_service()
        self.assertIsInstance(service.status, str)
        self.assertTrue(service.status)
        service.close()
        service.close()
        self.assertEqual(provider.close_calls, 1)
        with self.assertRaises(RuntimeError):
            service.reply("Hello", max_new_tokens=64)
        with self.assertRaises(RuntimeError):
            service.reset()
        self.assertEqual(provider.calls, [])

    def test_close_cancels_in_flight_reply_without_committing_it(self):
        class BlockingProvider(RecordingProvider):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.cancelled = threading.Event()

            def complete(self, messages, *, max_new_tokens, temperature):
                self.started.set()
                if not self.cancelled.wait(timeout=2):
                    raise TimeoutError("The test provider was not closed")
                return "An answer arriving after cancellation."

            def close(self):
                super().close()
                self.cancelled.set()

        provider = BlockingProvider()
        service = ChatService(provider=provider)
        self.addCleanup(service.close)
        failures = []

        def reply():
            try:
                service.reply("A slow question.", max_new_tokens=64)
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=reply, daemon=True)
        worker.start()
        self.assertTrue(provider.started.wait(timeout=2))
        service.close()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.assertEqual(service.history, [])
        self.assertEqual(provider.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
