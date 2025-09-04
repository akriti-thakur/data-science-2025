# run: streamlit run chatbot-app.py

# Imports
import streamlit as st
import nltk
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import wikipedia

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---- Chatbot logic ----

# Function to load file as sections (blocks separated by blank lines)
def load_sections(filepath):
    with open(filepath, 'r', errors='ignore') as f:
        raw = f.read()

    sections = [s.strip() for s in raw.split("\n\n") if s.strip()]
    return sections, raw.lower()

sections, raw_text = load_sections("panjabdcsa.txt")

# Tokenize sentences for precise Q&A
sent_tokens = nltk.sent_tokenize(raw_text)

# State trackers

lemmer = WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting logic
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
THANK_INPUTS = ("thanks", "thank you", "thank you so much")

def detect_intent(user_input):
    if user_input in THANK_INPUTS:
        return "thank_you"
    if user_input in GREETING_INPUTS:
        return "greeting"
    if user_input == "bye":
        return "exit"
    return "faq" 

# --- Wikipedia lookup ---
def wiki_answer(query):
    try:
        summary = wikipedia.summary(query, sentences=4)
        return "Here’s what I found on Wikipedia:\n\n" + summary
    except:
        return None

# --- File FAQ logic ---
def faq_answer(user_input):

    # --- Sentence-level similarity ---
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf_sent = TfidfVec.fit_transform(sent_tokens)
    vals_sent = cosine_similarity(tfidf_sent[-1], tfidf_sent)
    similarities_sent = vals_sent.flatten()
    similarities_sent[-1] = -1

    best_sent_idx = similarities_sent.argmax()
    best_sent_conf = similarities_sent[best_sent_idx]
    sent_tokens.remove(user_input)

    if best_sent_conf > 0.3:  # confident match
        return sent_tokens[best_sent_idx]

    # --- Section-level similarity ---
    sections.append(user_input)
    tfidf_sec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = tfidf_sec.fit_transform(sections)
    vals = cosine_similarity(tfidf[-1], tfidf)
    similarities = vals.flatten()
    similarities[-1] = -1

    idx = similarities.argmax()
    confidence = similarities[idx]
    sections.remove(user_input)

    if confidence > 0.1:
        return sections[idx]

    return None

# --- Response handler ---
def response(user_input):
    intent = detect_intent(user_input.lower())

    if intent == "greeting":
        return random.choice(["Hi!", "Hey!", "Hello! I’m glad you’re here."])
    elif intent == "thank_you":
        return "You're welcome!"
    elif intent == "faq":
        # Try file-based FAQ
        answer = faq_answer(user_input)
        if answer:
            return f"Here’s what I found:\n\n{answer}"

        # Fallback to Wikipedia
        wiki_info = wiki_answer(user_input)
        if wiki_info:
            return wiki_info

        return "Hmm... I’m not sure. Could you ask differently?"
    elif intent == "exit":
        return "Goodbye! Take care."
    else:
        return "I didn’t quite get that."
    

# ---- Streamlit UI ----
st.title("NLP DCSA Chatbot")
st.write("Ask the bot about DCSA, PU, or anything else (I’ll try Wiki too).")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True

# Input box 
if st.session_state.chat_active:
    user_input = st.chat_input("Type your message...")

    if user_input:
        bot_reply = response(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))

        if user_input.lower() == "bye":
            st.session_state.chat_active = False
else:
    st.info("The chat has ended. Refresh the page to start again.")

# Chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
