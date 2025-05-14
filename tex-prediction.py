# -*- coding: utf-8 -*-
# Advanced Call Center Prediction Console App using spaCy

import re
from collections import defaultdict
import random

# Step 0: Define sample call center corpus (Q&A)
corpus = [
    ("where is my order", "Your order is on the way."),
    ("what is the status of my delivery", "Your delivery is in transit."),
    ("i want to cancel my order", "Sure, we have initiated the cancellation."),
    ("how do i return a product", "You can return it via the returns section in your profile."),
    ("can i get a refund", "Yes, refunds are processed within 5-7 business days."),
    ("how long does delivery take", "Delivery takes 3-5 business days."),
    ("my product is damaged", "We're sorry! Please upload a photo and we will investigate."),
]

# Step 1: Preprocessing
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

# Step 2: Build Models
def build_models(corpus):
    intent_map = {}
    bigrams = defaultdict(lambda: defaultdict(int))
    trigrams = defaultdict(lambda: defaultdict(int))

    print("Building models from training data...")
    for q, a in corpus:
        words = tokenize(q)
        print(f"Training phrase: '{q}' → {words}")
        intent_map[tuple(words)] = a

        for i in range(len(words) - 1):
            bigrams[words[i]][words[i + 1]] += 1

        for i in range(len(words) - 2):
            trigrams[(words[i], words[i + 1])][words[i + 2]] += 1

    print("\nBigram Counts:")
    for w1, nexts in bigrams.items():
        print(f"{w1} → {dict(nexts)}")

    print("\nTrigram Counts:")
    for (w1, w2), nexts in trigrams.items():
        print(f"({w1}, {w2}) → {dict(nexts)}")

    return intent_map, bigrams, trigrams

# Step 3: Intent Matching
def find_best_match(user_input, intent_map):
    words = tokenize(user_input)
    print(f"\nTokenized user input: {words}")
    best_match = None
    best_score = 0

    for intent_words in intent_map:
        common = len(set(intent_words) & set(words))
        score = common / max(len(intent_words), len(words))
        print(f"→ Comparing to: {intent_words}, Score: {score:.2f}")
        if score > best_score:
            best_score = score
            best_match = intent_words

    if best_score > 0.5:
        print(f"Match found: {best_match} with score {best_score:.2f}")
        return intent_map[best_match]
    print("No strong match found.")
    return None

# Step 4: Fallback Prediction
def predict_next(words, bigrams, trigrams):
    if len(words) >= 2:
        pair = (words[-2], words[-1])
        if pair in trigrams:
            print(f"Trigram match for {pair}")
            next_words = trigrams[pair]
            sorted_words = sorted(next_words.items(), key=lambda x: -x[1])
            print(f"→ Candidates: {sorted_words}")
            return sorted_words[0][0]

    if len(words) >= 1:
        w = words[-1]
        if w in bigrams:
            print(f"Bigram match for {w}")
            next_words = bigrams[w]
            sorted_words = sorted(next_words.items(), key=lambda x: -x[1])
            print(f"→ Candidates: {sorted_words}")
            return sorted_words[0][0]

    print("No prediction available.")
    return None

# Step 5: Generate Fallback Sentence
def generate_fallback_sentence(seed, bigrams, trigrams, max_len=10):
    print(f"\nGenerating fallback sentence starting with: {seed}")
    sentence = seed[:]
    for _ in range(max_len):
        next_word = predict_next(sentence, bigrams, trigrams)
        if not next_word:
            break
        sentence.append(next_word)
    return " ".join(sentence)

# Step 6: Run Console App
def main():
    print("=== Advanced Call Center Text Prediction Console App ===\n")
    intent_map, bigrams, trigrams = build_models(corpus)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Have a good day!")
            break

        response = find_best_match(user_input, intent_map)
        if response:
            print(f"\nResponse: {response}")
        else:
            seed = tokenize(user_input)[:2]
            if not seed:
                print("Please enter a valid sentence.")
                continue
            generated = generate_fallback_sentence(seed, bigrams, trigrams)
            print(f"\nFallback Response: {generated}")

if __name__ == "__main__":
    main()
