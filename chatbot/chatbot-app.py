# run: streamlit run chatbot-app.py

# Imports
import streamlit as st
import nltk
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


last_idx = None


# ---- Chatbot logic ----
with open("panjabdcsa.txt", 'r', errors='ignore') as f:
    raw = f.read().lower()

sent_tokens= nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


lemmer = nltk.stem.WordNetLemmatizer()
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
    if "tell me more" in user_input:
        return "tell_me_more"
    if user_input == "bye":
        return "exit"
    return "faq" 

def faq_answer(user_input):
    global last_idx
    sent_tokens.append(user_input)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = TfidfVec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf)
    similarities = vals.flatten()
    similarities[-1] = -1  # ignore self-match

    idx = similarities.argmax()
    confidence = similarities[idx]

    sent_tokens.remove(user_input)

    if confidence == 0:
        last_idx = None
        return None
    else:
        last_idx = idx
        return sent_tokens[idx]

    
def tell_me_more(threshold=0.2):
    global last_idx
    if last_idx is not None and last_idx + 1 < len(sent_tokens):
        # Compare current sentence with next one
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
        tfidf = TfidfVec.fit_transform([sent_tokens[last_idx], sent_tokens[last_idx + 1]])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        if similarity >= threshold:
            last_idx += 1
            return sent_tokens[last_idx]
        else:
            return None
    else:
        return None


def response(user_input):
    intent = detect_intent(user_input.lower())
    if intent == "greeting":
        return random.choice(["Hi!", "Hey!", "Hello! I’m glad you’re here."])
    elif intent == "thank_you":
        return "You're welcome!"
    elif intent == "tell_me_more":
        more_info = tell_me_more()
        return f"Sure! Here's more: {more_info}" if more_info else "I don’t have more details right now."
    elif intent == "faq":
        answer = faq_answer(user_input)
        return f"Here’s what I found: {answer}" if answer else "Hmm... I’m not sure. Could you ask differently?"
    elif intent == "exit":
        return "Goodbye! Take care."
    else:
        return "I didn’t quite get that."
    


# ---- Streamlit UI ----
st.title("NLP DCSA Chatbot")
st.write("Ask the bot about DCSA,PU.")

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

        # If user said "bye", stop the chat
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
