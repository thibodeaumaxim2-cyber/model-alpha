import tempfile
import unittest
from pathlib import Path

import numpy as np

from alpha import Conversation, FALLBACK, Network, encode, load_data, train


class AlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.network = train(cls.data, verbose=False)

    def test_held_out_messages(self):
        examples = [(text, label) for label, intent in self.data.items() for text in intent['test']]
        correct = sum(self.network.predict(text)[0] == label for text, label in examples)
        self.assertGreaterEqual(correct / len(examples), 0.85)

    def test_saved_weights_preserve_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            self.network.save(path)
            restored = Network.load(path)
            for text in ['hello', 'explain machine learning', 'xyzzy plugh']:
                self.assertEqual(self.network.predict(text), restored.predict(text))
            self.assertEqual(self.network.responses, restored.responses)

    def test_unknown_and_empty_messages(self):
        bot = Conversation(self.network)
        self.assertEqual(bot.reply('xyzzy plugh'), FALLBACK)
        self.assertEqual(bot.reply('calculate quantum chromodynamics'), FALLBACK)
        self.assertIn('Type a message', bot.reply('  '))
        self.assertIn('1,000', bot.reply('x' * 1001))

    def test_name_memory_reset_and_session_isolation(self):
        bot = Conversation(self.network)
        self.assertIn('know your name', bot.reply('what is my name?'))
        self.assertIn('Sam', bot.reply('My name is Sam.'))
        self.assertEqual(bot.reply('what is my name?'), 'Your name is Sam.')
        self.assertIn('Sam', bot.reply('hello'))
        self.assertIsNone(Conversation(self.network).name)
        bot.reply('/reset')
        self.assertIsNone(bot.name)

    def test_follow_up_and_context_expiration(self):
        bot = Conversation(self.network, seed=42)
        self.assertIn('joke?', bot.reply('I am sad'))
        self.assertIn(bot.reply('yes please'), self.network.responses['joke'])
        self.assertFalse(bot.pending_joke)
        bot.reply('I am sad')
        self.assertIn('Okay', bot.reply('no thanks'))
        bot.reply('I am sad')
        bot.reply('what is python')
        self.assertFalse(bot.pending_joke)
        bot.reply('I am sad')
        bot.reply('/reset')
        self.assertFalse(bot.pending_joke)

    def test_feature_normalization(self):
        vocabulary = self.network.vocabulary
        np.testing.assert_array_equal(encode('HELLO!', vocabulary), encode('hello', vocabulary))
        self.assertAlmostEqual(np.linalg.norm(encode('hello alpha', vocabulary)), 1.0)
        self.assertEqual(np.linalg.norm(encode('xyzzy', vocabulary)), 0)


if __name__ == '__main__':
    unittest.main()
