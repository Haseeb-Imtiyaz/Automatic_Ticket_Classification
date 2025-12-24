import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import en_core_web_sm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Load spaCy model
nlp = en_core_web_sm.load()
stopwords = nlp.Defaults.stop_words

def clean_text(text):
    """Clean the text by removing unwanted characters and patterns."""
    text = text.lower()    # lower case
    text = text.replace("chase", "")              # removing chase word
    text = re.sub('\[\S+\]', '', text).strip()    # remove text in square bracket
    text = text.translate(str.maketrans('', '', string.punctuation))     # remove punctuation
    text = re.sub('\S*\d\S*', '', text).strip()   # remove words containing the number
    text = re.sub(r'\b\w*xx\w*\b', '', text).strip()  # remove PII data
    return text.strip()

def lemmatizer(text):
    """Lemmatize text and remove stopwords."""
    lemma = WordNetLemmatizer().lemmatize
    return ' '.join([lemma(word) for word in text.split() if word.lower() not in set(stopwords)])

def extract_pos(text):
    """Extract POS tags (VB for Verb, NN for Noun)."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    sent = [word for word, tag in tagged if tag.startswith('VB') or tag.startswith('NN')]
    return ' '.join(sent)

def preprocess_text(text):
    """Complete preprocessing pipeline."""
    if not text or not text.strip():
        return ""
    
    # Clean text
    cleaned = clean_text(text)
    
    if not cleaned:
        return ""
    
    # Lemmatize
    lemmatized = lemmatizer(cleaned)
    
    if not lemmatized:
        return ""
    
    # Extract POS
    pos_text = extract_pos(lemmatized)
    
    return pos_text

