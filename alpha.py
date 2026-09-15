"""A small NumPy neural chatbot. Run `python alpha.py --help` to begin."""

import argparse
import json
from pathlib import Path
import random
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "intents.json"
MODEL = ROOT / "model.json"
FALLBACK = "I don't recognize that topic yet. Try asking about AI, Python, my training, or a joke."


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def features(text):
    words = tokens(text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def encode(text, vocabulary):
    vector = np.zeros(len(vocabulary))
    for feature in features(text):
        if feature in vocabulary:
            vector[vocabulary[feature]] = 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def load_data(path=DATA):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or len(data) < 2:
        raise ValueError("Provide at least two intents in intents.json.")
    seen = set()
    for label, intent in data.items():
        for field in ("train", "test", "responses"):
            values = intent.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"{label}.{field} must contain nonempty strings.")
        for text in intent["train"] + intent["test"]:
            key = tuple(tokens(text))
            if not key or key in seen:
                raise ValueError(f"Empty or duplicate example: {text!r}")
            seen.add(key)
    return data


class Network:
    def __init__(self, vocabulary, labels, responses, weights):
        self.vocabulary = vocabulary
        self.labels = labels
        self.responses = responses
        self.w1, self.b1, self.w2, self.b2 = weights

    def predict(self, text):
        x = encode(text, self.vocabulary)
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top = np.argsort(probabilities)[-2:]
        label = self.labels[top[-1]]
        confidence = float(probabilities[top[-1]])
        margin = confidence - float(probabilities[top[-2]])
        words = tokens(text)
        coverage = sum(word in self.vocabulary for word in words) / max(1, len(words))
        # These are heuristics, not a guarantee against confident mistakes.
        accepted = bool(words) and coverage >= 0.6 and confidence >= 0.65 and margin >= 0.25
        return label, confidence, accepted

    def save(self, path=MODEL):
        payload = {
            "version": 1,
            "vocabulary": self.vocabulary,
            "labels": self.labels,
            "responses": self.responses,
            "weights": [w.tolist() for w in (self.w1, self.b1, self.w2, self.b2)],
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path=MODEL):
        payload = json.loads(Path(path).read_text())
        if payload.get("version") != 1:
            raise ValueError("Unsupported model version. Run 'python alpha.py train'.")
        return cls(payload["vocabulary"], payload["labels"], payload["responses"],
                   [np.asarray(w, dtype=float) for w in payload["weights"]])


def train(data, epochs=800, seed=42, verbose=True):
    labels = sorted(data)
    examples = [(text, index) for index, label in enumerate(labels)
                for text in data[label]["train"]]
    vocabulary = {word: index for index, word in enumerate(
        sorted({word for text, _ in examples for word in features(text)}))}
    x = np.stack([encode(text, vocabulary) for text, _ in examples])
    y = np.array([label for _, label in examples])
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(1 / len(vocabulary)), (len(vocabulary), 48))
    b1 = np.zeros(48)
    w2 = rng.normal(0, np.sqrt(1 / 48), (48, len(labels)))
    b2 = np.zeros(len(labels))
    weights = [w1, b1, w2, b2]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]

    for step in range(1, epochs + 1):
        # Forward pass: word features -> hidden layer -> intent probabilities.
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean()

        # Backpropagation: derivatives of cross-entropy and tanh.
        output_grad = probabilities.copy()
        output_grad[np.arange(len(y)), y] -= 1
        output_grad /= len(y)
        hidden_grad = (output_grad @ w2.T) * (1 - hidden ** 2)
        gradients = [x.T @ hidden_grad, hidden_grad.sum(axis=0),
                     hidden.T @ output_grad, output_grad.sum(axis=0)]

        # Adam: keep running estimates of gradient size and direction.
        for index, (weight, gradient) in enumerate(zip(weights, gradients)):
            first[index] = 0.9 * first[index] + 0.1 * gradient
            second[index] = 0.999 * second[index] + 0.001 * gradient ** 2
            corrected_first = first[index] / (1 - 0.9 ** step)
            corrected_second = second[index] / (1 - 0.999 ** step)
            weight -= 0.01 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if verbose and (step == 1 or step % 200 == 0 or step == epochs):
            print(f"Epoch {step:4d}/{epochs} | training loss {loss:.4f}")

    return Network(vocabulary, labels, {label: data[label]["responses"] for label in labels}, weights)


class Conversation:
    """Memory and follow-ups are explicit rules, separate from learned weights."""

    def __init__(self, network, seed=None):
        self.network = network
        self.rng = random.Random(seed)
        self.name = None
        self.pending_joke = False

    def reply(self, text):
        text = text.strip()
        if text == "/reset":
            self.name = None
            self.pending_joke = False
            return "Memory cleared. Let's start again."
        if not text:
            return "Type a message to start chatting."
        if len(text) > 1000:
            return "Please keep messages under 1,000 characters."
        normalized = " ".join(tokens(text))
        if normalized in {"what is my name", "whats my name", "do you remember my name"}:
            return f"Your name is {self.name}." if self.name else "I don't know your name yet. Say 'My name is Sam'."
        match = re.fullmatch(r"(?:my name is|call me)\s+([\w'-]+(?: [\w'-]+){0,2})[.!]?", text, re.I)
        if match:
            self.name = match.group(1)[:60]
            return f"Nice to meet you, {self.name}! I'll remember your name until /reset or exit."
        if self.pending_joke and normalized in {"yes", "yes please", "sure", "okay", "no", "no thanks"}:
            self.pending_joke = False
            if normalized.startswith("no"):
                return "Okay. We can talk about AI or Python instead."
            return self.rng.choice(self.network.responses["joke"])
        self.pending_joke = False
        label, _, accepted = self.network.predict(text)
        if not accepted:
            return FALLBACK
        self.pending_joke = label == "sad" and "joke" in self.network.responses
        response = self.rng.choice(self.network.responses[label])
        return response.replace("{name}", f", {self.name}" if self.name else "")


def evaluate(network, data):
    total = correct = accepted = accepted_correct = 0
    for expected, intent in data.items():
        for text in intent["test"]:
            predicted, confidence, ok = network.predict(text)
            total += 1
            correct += predicted == expected
            accepted += ok
            accepted_correct += ok and predicted == expected
            if predicted != expected or not ok:
                print(f"  {text!r}: expected={expected}, predicted={predicted}, "
                      f"score={confidence:.2f}, accepted={ok}")
    print(f"Held-out intent accuracy: {correct}/{total} ({correct / total:.1%})")
    print(f"Accepted messages: {accepted}/{total}; correct accepted replies: {accepted_correct}/{total}")
    print("Small, hand-written test set: this does not measure open-ended chat ability.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "chat", "evaluate"], default="chat")
    parser.add_argument("--epochs", type=int, default=800)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.command == "train" or not MODEL.exists():
        print("Training Alpha on intents.json...")
        network = train(load_data(), epochs=args.epochs)
        network.save()
        print(f"Saved learned weights to {MODEL}")
    else:
        network = Network.load()
    if args.command in {"train", "evaluate"}:
        evaluate(network, load_data())
        return
    conversation = Conversation(network)
    print("Alpha: Hello! I'm a small learning chatbot. Type /quit to exit or /reset to clear memory.")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAlpha: Goodbye!")
            break
        if text.strip().lower() in {"/quit", "/exit"}:
            print("Alpha: Goodbye!")
            break
        print("Alpha:", conversation.reply(text))


if __name__ == "__main__":
    main()
