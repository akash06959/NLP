# ==============================================================================
# 1. SETUP: INSTALL AND IMPORT LIBRARIES
# ==============================================================================
# Install required libraries silently
!pip install gensim pyLDAvis==3.4.1 nltk -q

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from sklearn.datasets import fetch_20newsgroups
import warnings

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download necessary NLTK data (only needs to be done once per environment)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)


# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
# Define the newsgroup categories we want to analyze
categories = ['rec.sport.baseball', 'sci.electronics', 'talk.politics.guns', 'comp.graphics']

# Fetch the dataset, removing headers, footers, and quotes for cleaner text
newsgroups_data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')
)

# Extract the text content from the dataset
corpus_text = newsgroups_data.data
print(f"Successfully loaded {len(corpus_text)} documents.")


# ==============================================================================
# 3. PREPROCESS TEXT
# ==============================================================================
# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and prepares text data for topic modeling.
    - Tokenizes text into words.
    - Removes stopwords, short tokens (< 4 chars).
    - Lemmatizes tokens to their base form.
    """
    # Use gensim's simple_preprocess for robust tokenization
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    
    # Lemmatize and filter out stopwords and short words
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 3:
            # Lemmatize verbs, as they often carry significant topic meaning
            processed_tokens.append(lemmatizer.lemmatize(token, pos='v'))
            
    return processed_tokens

# Apply the preprocessing function to each document in the corpus
print("Preprocessing text data...")
processed_corpus = [preprocess_text(doc) for doc in corpus_text]
print("Preprocessing complete.")
print("\n--- Example of a processed document ---")
print(processed_corpus[0])


# ==============================================================================
# 4. CREATE DICTIONARY AND CORPUS FOR LDA
# ==============================================================================
# Create a dictionary mapping unique words to IDs
id2word = Dictionary(processed_corpus)

# Filter out words that are too rare or too common to improve topic quality
# no_below=15: ignore tokens that appear in less than 15 documents
# no_above=0.5: ignore tokens that appear in more than 50% of documents
id2word.filter_extremes(no_below=15, no_above=0.5)

# Create a Bag-of-Words (BoW) corpus
# This converts each document into a list of (word_id, frequency) tuples
bow_corpus = [id2word.doc2bow(doc) for doc in processed_corpus]
print(f"\nNumber of unique tokens in dictionary after filtering: {len(id2word)}")


# ==============================================================================
# 5. BUILD AND TRAIN THE LDA MODEL
# ==============================================================================
# Set the number of topics to discover
NUM_TOPICS = 4

# Build the LDA model using LdaMulticore for faster, parallelized training
print("\nTraining LDA model...")
lda_model = LdaMulticore(
    corpus=bow_corpus,
    id2word=id2word,
    num_topics=NUM_TOPICS,
    random_state=100,
    chunksize=100,
    passes=10,
    workers=2  # Number of CPU cores to use
)
print("LDA model training complete.")

# Print the keywords for each of the 4 topics
print("\n--- Identified Topics and Their Top Words ---")
topics = lda_model.print_topics(num_words=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {topic[1]}")


# ==============================================================================
# 6. VISUALIZE THE TOPICS
# ==============================================================================
# Enable notebook mode for pyLDAvis
pyLDAvis.enable_notebook()

# Prepare the data for the interactive visualization
vis_data = gensimvis.prepare(lda_model, bow_corpus, id2word)

# Display the visualization
print("\n--- Generating Interactive Topic Visualization ---")
vis_data