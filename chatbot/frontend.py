# run: streamlit run frontend.py

import streamlit as st
import import_ipynb
import chatbot
from chatbot import response

st.title("NLP DCSA Chatbot")
st.write("Ask the bot about DCSA or PU.")

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