import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trulens.core import TruSession
from openai import OpenAI
import torch.nn as nn

#session = TruSession()
#session.reset_database()

import uuid
# 1. Import TruChain from the core package
#from trulens.apps.langchain import TruChain
from trulens.core import Tru
from trulens.dashboard import run_dashboard
from trulens.core import TruSession
from trulens.apps.app import TruApp
from langchain_openai import ChatOpenAI
import rag
from retrieval_exp import vector_store
from eval import f_groundedness, f_answer_relevance, f_context_relevance#, f_debug_context
import streamlit as st
import news

#tru = Tru()

@st.cache_resource
def setup_application_engine():
    """Initializes LLM, RAG App, and TruLens Recorder exactly ONCE."""

    session = TruSession()
    session.reset_database()
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o",          
        temperature=0            
    )

    # Instantiate your real RAG application using the imported vector_store
    rag_app = rag.ConversationalRAG(vector_store, llm, memory_store={})
    
    # Initialize TruLens Recorder here so it doesn't duplicate on every rerun
    tru_recorder = TruApp(
        app=rag_app,
        app_name="Sustainability_RAG",
        app_version="v2",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
        #feedbacks=[f_debug_context],
        connector=session.connector
    )
    print("Feedbacks:")
    #print(f_groundedness)
    #print(f_answer_relevance)
    #print(f_context_relevance)
   
    # Launch dashboard once
    print("🚀 Launching TruLens Dashboard...")
    run_dashboard(session)
    
    # Return BOTH so we can use them in the UI loop
    return rag_app, tru_recorder

# Fetch our cached engine components
rag_engine, tru_recorder = setup_application_engine()


# ==========================================
# VISUAL STREAMLIT PAGE
# ==========================================
st.set_page_config(page_title="Sustainability AI Assistant", page_icon="🌱")

# STATE MANAGEMENT
# Keeps your user's specific session ID active across reruns
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Keeps the chat UI history alive across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "recommended_articles" not in st.session_state:
    st.session_state.recommended_articles = []  # Holds list of article dicts


# ==========================================
# SIDEBAR RECOMMENDER SYSTEM UI
# ==========================================
with st.sidebar:
    st.header("✨ Smart Reading List")
    st.caption("AI-extracted news based on your recent conversation topics.")
    
    # Only allow extraction if conversation has started
    if not st.session_state.chat_history:
        st.info("Start chatting with the assistant to unlock contextual news feeds!")
    else:
        if st.button("🔄 Refresh Topic Recommendations", use_container_width=True, type="primary"):
            with st.spinner("Running topic extraction..."):
                # 1. Parse your user chat history tracking stack backwards
                user_questions = news.chat_history_to_user_questions(st.session_state.chat_history)
                
                if user_questions:
                    # 2. Call your adjusted function wrapper directly with the list
                    articles = news.suggest_news_article(user_questions)
                    
                    # 3. Cache it securely inside session state
                    st.session_state.recommended_articles = articles
                else:
                    st.warning("No user queries found to analyze yet.")
                    
    st.markdown("---")

    # Render Card Interfaces for Articles
    if st.session_state.recommended_articles:
        for article in st.session_state.recommended_articles:
            st.markdown(
                f"""
                <div style="border: 1px solid #4A5568; padding: 12px; border-radius: 8px; margin-bottom: 12px; background-color: rgba(255,255,255,0.02);">
                    <h4 style="margin: 0 0 6px 0; color: #4FFFB0; font-size: 13px; line-height:1.3;">📍 {article['title']}</h4>
                    <p style="font-size: 11px; color: #CBD5E1; margin: 0 0 8px 0; line-height: 1.4;">
                        {article['snippet'][:130]}...
                    </p>
                    <a href="{article['url']}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #1E293B; border: 1px solid #4FFFB0; color: #4FFFB0; padding: 4px 8px; border-radius: 4px; font-size: 10px; cursor: pointer;">
                            Read Article ↗
                        </button>
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
            )
    elif st.session_state.chat_history:
        st.caption("Click the refresh button above to populate news related to your current topics.")


# ==========================================
# CENTRAL CHATBOT INTERFACE
# ==========================================

st.title("🌱 Sustainability AI Assistant")
st.caption("Context-Aware Decision Support Engine For People That Want Clarity On How To Contribute to Sustainability")


# ==========================================
#     RENDER EXISTING PAST CHAT BUBBLES
# ==========================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
#        HANDLE NEW USER SUBMISSIONS
# ==========================================
if user_question := st.chat_input("Ask environmental or climate related questions..."):
    
    # 1. Instantly render the user's question on screen
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # 2. Process the query inside the assistant UI block
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data repositories..."):
            try:
                # Wrap the execution block with your cached TruLens recorder
                with tru_recorder as recording:
                    response = rag_engine(user_input=user_question, session_id=st.session_state.session_id)
                #record = recording.get()

                #print(record.calls.keys())
                print(type(tru_recorder))
                print(dir(tru_recorder))
                print("TRU RECORDER CREATED")

                try:
                    print("feedbacks =", tru_recorder.feedbacks)
                except Exception as e:
                    print("No feedback attribute:", e)
                # Display the real response from your pipeline
                st.markdown(response)
                
            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)
            
    # Commit agent update back to persistent state layout
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Force a component refresh to make sure data frames match instantly
    st.rerun()
