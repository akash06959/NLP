import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample Kerala-related news text
text = ("The voice recorder was in the Boeing 787 aircraft’s second black box, which the Indian authorities said they had found on Sunday. The first, containing the flight data recorder, was located within 28 hours of Thursday’s disaster in Ahmedabad, in which at least 279 people died.")

# Process the text
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("1. Tokens:")
print(tokens)

# Stop-word removal
non_stop_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
print("\n2. After Stop-word Removal:")
print(non_stop_tokens)

# Lemmatization (excluding stopwords & punctuation)
lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("\n3. Lemmatized Words:")
print(lemmas)
