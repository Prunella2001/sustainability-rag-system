#=====================================================================================
#                          RETRIEVAL AUGMENTED GENERATION                                      
#                     This module creates the RAG class itself                   
#=====================================================================================

import uuid
from langchain_core.runnables.history import RunnableWithMessageHistory
#from trulens.core import instrument
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
import retrieval_exp

#This is where we set up the weights for the retrieval experiment
weights = [
        [0.0, 1.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [1.0, 0.0],
        [0.3, 0.7],
        [0.4, 0.6]
    ]
# This dictionary will hold the history for all active chat sessions

def reorder_documents(docs: list) -> list:
    """Sorts docs so the highest-ranked elements sit at the boundaries."""
    if not docs:
        return []
    # If using LangChain's reranker, it automatically orders them highest to lowest score
    sorted_docs = list(docs) 
    reordered = [None] * len(sorted_docs)
    left, right = 0, len(sorted_docs) - 1
    
    for i, doc in enumerate(sorted_docs):
        if i % 2 == 0:
            reordered[left] = doc
            left += 1
        else:
            reordered[right] = doc
            right -= 1
    return reordered

#class ConversationalRAG:
#    def __init__(self, vector_store, llm, memory_store):
#        self.vector_store = vector_store
#        self.llm = llm
#        self.memory_store = memory_store # e.g., a dict to hold ChatMessageHistory objects
#
#        # Initializing advanced compression retriever components
#        self.compression_retriever = self._setup_advanced_retriever()
#
#        # Building final prompt generation chains
#        self.chain = self._setup_chain()
#
#    def _setup_advanced_retriever(self):
#
#        """Assembles the hybrid ensemble + cross-encoder reranking pipeline."""
#
#        return retrieval_exp.found_retriever(weights)
#
#    # 1. FLAG YOUR ADVANCED RETRIEVER SUB-METHOD FOR OTEL
#    @instrument(
#        span_type=SpanAttributes.SpanType.RETRIEVAL,
#        attributes={
#            SpanAttributes.RETRIEVAL.QUERY_TEXT: "user_input",
#            # This tells TruLens that the array returned here is the formal context
#            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
#        },
#    )
#    def get_context(self, user_input: str) -> list:
#        """
#        Executes retrieval + compression/reranking.
#        TruLens records this method, allowing you to access the final 3 document chunks.
#        """
#        # Returns a list of LangChain Document objects
#        docs = self.compression_retriever.invoke(user_input)
#        return [doc.page_content for doc in docs]
#        #return [ doc.metadata.get("source") for doc in self.compression_retriever.invoke(user_input)]
#
#    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
#      if session_id not in self.memory_store:
#          self.memory_store[session_id] = InMemoryChatMessageHistory()
#      return self.memory_store[session_id]
#
##    def _setup_chain(self):
##        # wrapped history chain
##        rephrase_prompt = ChatPromptTemplate.from_messages([
##        MessagesPlaceholder(variable_name="chat_history"),
##        ("user", "{input}"),
##        ("user", "Based on the conversation above, rephrase the user's question so that it stands on its own and is understandable to a search engine.")
##        ])
##
##        # history aware compression_retriever
##        history_aware_retriever = create_history_aware_retriever(
##            self.llm,
##            self.compression_retriever,
##            rephrase_prompt
##        )
##
##        #La chaîne de génération de réponse (Prompt final + LLM)
##        #qa_prompt = ChatPromptTemplate.from_messages([
##        #    ("system", """You are an AI assistant specializing in sustainability. Answer the question based only on the following context :\n\n{context}. If you cannot
##        #      find the answer in the context, say "Sorry Sustainability Warrior, I don't have enough information to answer this question."""),
##        #    MessagesPlaceholder(variable_name="chat_history"),
##        #    ("user", "{input}"),
##        #])
##        qa_prompt = ChatPromptTemplate.from_messages([
##            ("system", """You are an expert sustainability systems analyst. 
##
##            Your task is to answer the user's question using ONLY the provided text blocks. 
##
##            CRITICAL OPERATIONAL RULES:
##            1. Grounding: Every claim, definition, or metric you provide MUST be directly traced to a source URL present in the context.
##            2. Citation Format: Append the source URL directly to the sentence it supports (e.g., "Plastic takes 450 years to break down [Source: https://www.epa.gov]").
##            3. Strict Fallback: If the text blocks do not contain undeniable evidence to answer the prompt, state exactly: "Sorry Sustainability Warrior, I don't have enough information to answer this question." Do not attempt to use background knowledge.
##            4. Transparency: Never state a fact that lacks an explicitly attached source link in the context below.
##
##            CONTEXT DOCUMENTS:
##            \n\n{context}"""),
##                MessagesPlaceholder(variable_name="chat_history"),
##                ("user", "{input}"),
##            ])
##
##        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
##
##        #La fusion finale (Conversational RAG Chain)
##        conversational_rag_chain = create_retrieval_chain(
##            history_aware_retriever,
##            question_answer_chain
##        )
##
##        # Wrapping the existing conversational_rag_chain
##        conversational_rag_w_history = RunnableWithMessageHistory(
##            conversational_rag_chain,
##            self.get_session_history,
##            input_messages_key="input",
##            history_messages_key="chat_history",
##            output_messages_key="answer"
##        )
##        return conversational_rag_w_history
#
#    def _setup_chain(self):
#        # 1. Clear prompt for turning conversational follow-ups into standalone queries
#        rephrase_prompt = ChatPromptTemplate.from_messages([
#            MessagesPlaceholder(variable_name="chat_history"),
#            ("user", "{input}"),
#            ("user", "Based on the conversation above, rephrase the user's question so that it stands on its own and is understandable to a search engine.")
#        ])
#
#        # A small internal chain that runs the rephrasing prompt through the LLM and casts to string
#        condense_question_chain = rephrase_prompt | self.llm | str
#
#        # 2. Your Expert Analyst Prompt Template (with strict grounding rules)
#        qa_prompt = ChatPromptTemplate.from_messages([
#            ("system", """You are an expert sustainability systems analyst. 
#    
#                Your task is to answer the user's question using ONLY the provided text blocks. 
#
#                CRITICAL OPERATIONAL RULES:
#                1. Grounding: Every claim, definition, or metric you provide MUST be directly traced to a source URL present in the context.
#                2. Strict Fallback: If the text blocks do not contain undeniable evidence to answer the prompt, state exactly: "Sorry Sustainability Warrior, I don't have enough information to answer this question." Do not attempt to use background knowledge.
#                3. Transparency: Never state a fact that lacks an explicitly attached source link in the context below.
#
#                CONTEXT DOCUMENTS:
#                \n\n{context}"""),
#                            MessagesPlaceholder(variable_name="chat_history"),
#                            ("user", "{input}"),
#                        ])
#    
#        # Helper function to decide whether to rephrase the question based on history presence
#        def route_input(inputs: dict):
#            if not inputs.get("chat_history"):
#                return inputs["input"]
#            return condense_question_chain
#
#        # 3. Explicitly construct the core conversational RAG pipeline using clear dictionary mapping
#        # This securely executes your compression retriever and reorders chunks before context injection
#        conversational_rag_chain = {
#            "context": lambda x: reorder_documents(self.compression_retriever.invoke(route_input(x))),
#            "chat_history": lambda x: x.get("chat_history", []),
#            "input": lambda x: x["input"]
#        } | qa_prompt | self.llm
#
#        # 4. Wrap the execution layout inside your session history manager
#        conversational_rag_w_history = RunnableWithMessageHistory(
#            conversational_rag_chain,
#            self.get_session_history,
#            input_messages_key="input",
#            history_messages_key="chat_history",
#        )
#        return conversational_rag_w_history
#
#    @instrument(
#        span_type=SpanAttributes.SpanType.RECORD_ROOT,
#        attributes={
#            SpanAttributes.RECORD_ROOT.INPUT: "user_input",
#            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
#        },
#    )
#    def __call__(self, user_input: str, session_id: str) -> str:
#        """
#        The main execution point that TruLens wraps.
#        """
#        config = {"configurable": {"session_id": session_id}}
#
#        _ = self.get_context(user_input)
#        # Invoke your internal chain
#        response = self.chain.invoke({"input": user_input}, config=config)
#
#        # Return just the text answer so TruLens can map it cleanly
#        if hasattr(response, 'content'):
#            return response.content
#        elif isinstance(response, dict):
#            return response.get("answer", str(response))
#        #response.get("answer", response)
#        return response

class ConversationalRAG:
    def __init__(self, vector_store, llm, memory_store):
        self.vector_store = vector_store
        self.llm = llm
        self.memory_store = memory_store 
        self.compression_retriever = self._setup_advanced_retriever()
        self.chain = self._setup_chain()

    def _setup_advanced_retriever(self):
        return retrieval_exp.found_retriever(weights)

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.memory_store:
            self.memory_store[session_id] = InMemoryChatMessageHistory()
        return self.memory_store[session_id]

    def _setup_chain(self):
        """Chaîne secondaire pour condenser la question si historique présent."""
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Based on the conversation above, rephrase the user's question so that it stands on its own and is understandable to a search engine.")
        ])
        return rephrase_prompt | self.llm | str

    def _get_qa_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert sustainability systems analyst. 
Your task is to answer the user's question using ONLY the provided text blocks. 

CRITICAL OPERATIONAL RULES:
1. Grounding: Every claim, definition, or metric you provide MUST be directly traced to a source URL present in the context.
2. Citation Format: Append the source URL directly to the sentence it supports (e.g., "Plastic takes 450 years to break down [Source: https://www.epa.gov]").
3. Strict Fallback: If the text blocks do not contain undeniable evidence to answer the prompt, state exactly: "Sorry Sustainability Warrior, I don't have enough information to answer this question." Do not attempt to use background knowledge.
4. Transparency: Never state a fact that lacks an explicitly attached source link in the context below.

CONTEXT DOCUMENTS:
\n\n{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

    # =====================================================================================
    #  JALON 1 : RETRIEVAL (Cible explicitement l'argument 'query' et le retour)
    # =====================================================================================
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        """Exécute le retriever hybride, applique le reordering et extrait le texte brut."""
        docs = self.compression_retriever.invoke(query)
        reordered = reorder_documents(docs)
        # On extrait uniquement les chaînes de texte pour l'évaluateur de TruLens
        return [doc.page_content for doc in reordered]

    # =====================================================================================
    #  JALON 2 : GENERATION
    # =====================================================================================
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_list: list, history: list) -> str:
        """Prend le texte tracé et l'envoie au LLM via ton prompt strict."""
        if not context_list:
            return "Sorry Sustainability Warrior, I don't have enough information to answer this question."

        context_block = "\n\n".join(context_list)
        payload = {
            "context": context_block,
            "chat_history": history,
            "input": query
        }
        
        qa_prompt = self._get_qa_prompt()
        response = (qa_prompt | self.llm).invoke(payload)
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    # =====================================================================================
    #  JALON 3 : ROOT EXECUTION (L'entrée principale observée par TruApp)
    # =====================================================================================
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str, session_id: str) -> str:
        # A. Récupération manuelle de l'historique
        history_store = self.get_session_history(session_id)
        chat_history = history_store.messages

        # B. Rephrasage contextuel si nécessaire
        target_query = query
        if len(chat_history) > 0:
            target_query = self.chain.invoke({"input": query, "chat_history": chat_history})

        # C. Appel de l'étape de Retrieval tracée
        context_list = self.retrieve(query=target_query)

        # D. Appel de l'étape de Generation tracée
        answer = self.generate_completion(query=target_query, context_list=context_list, history=chat_history)

        # E. Mise à jour de la mémoire de session
        history_store.add_user_message(query)
        history_store.add_ai_message(answer)

        return answer

    def __call__(self, user_input: str, session_id: str) -> str:
        return self.query(query=user_input, session_id=session_id)