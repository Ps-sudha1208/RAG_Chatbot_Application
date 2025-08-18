# main.py
import os
from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# from langchain_huggingface import (
#     HuggingFaceEmbeddings,
#     ChatHuggingFace,
#     HuggingFaceEndpoint,
# )
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser


# class ChatBot:
#     def __init__(self, index_name: str = "langchain-demo", cloud: str = "aws", region: str = "us-east-1"):
#         load_dotenv()

#         # HF env compatibility
#         if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None and os.getenv("HUGGINGFACEHUB_API_KEY"):
#             os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_KEY")

#         pinecone_key = os.getenv("PINECONE_API_KEY")
#         hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#         if not pinecone_key or not hf_token:
#             raise RuntimeError("Missing PINECONE_API_KEY or HUGGINGFACEHUB_API_TOKEN in environment.")

#         # ------- Load & split docs -------
#         if not os.path.exists("./horoscope.txt"):
#             raise FileNotFoundError("Couldn't find './horoscope.txt'. Make sure the file exists.")

#         loader = TextLoader("./horoscope.txt")
#         documents = loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
#         self.docs = text_splitter.split_documents(documents)

#         # ------- Embeddings -------
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         dim = len(self.embeddings.embed_query("ping"))  # expected 768

#         # ------- Pinecone setup -------
#         pc = Pinecone(api_key=pinecone_key)
#         self.index_name = index_name

#         # Handle both dict and object forms from pc.list_indexes() across client versions
#         def _idx_name(x):
#             return x.name if hasattr(x, "name") else (x.get("name") if isinstance(x, dict) else None)

#         existing = {_idx_name(i) for i in pc.list_indexes()}
#         if self.index_name not in existing:
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=dim,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud=cloud, region=region),
#             )

#         idx = pc.Index(self.index_name)
#         stats = idx.describe_index_stats()

#         # Total vectors across namespaces
#         namespaces = stats.get("namespaces", {}) or {}
#         total = sum(ns.get("vector_count", 0) for ns in namespaces.values())

#         # ------- Vector store -------
#         if total == 0:
#             self.vectorstore = PineconeVectorStore.from_documents(
#                 self.docs, embedding=self.embeddings, index_name=self.index_name
#             )
#         else:
#             self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

#         # Optional quick sanity check (safe even if no results)
#         try:
#             res = self.vectorstore.similarity_search("lucky number for leo", k=3)
#             for i, r in enumerate(res, 1):
#                 # print(f"[{i}] {r.page_content[:200]}...")
#                 pass
#         except Exception:
#             pass

#         self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

#         # ------- LLM -------
#         self.chat_llm = ChatHuggingFace(
#             llm=HuggingFaceEndpoint(
#                 repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
#                 task="text-generation",
#                 temperature=0.7,
#                 top_k=50,
#                 max_new_tokens=256,
#                 do_sample=True,
#                 return_full_text=False,
#                 huggingfacehub_api_token=hf_token,
#             )
#         )

#         # ------- Prompt -------
#         prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a fortune teller. Use the provided context to answer. "
#              "If you don't know, say you don't know. Keep the answer within 2 sentences and concise."),
#             ("human", "Context:\n{context}\n\nQuestion: {question}")
#         ])

#         # Keep as a nested function to avoid capturing self in RunnableLambda
#         def format_docs(docs):
#             return "\n\n".join(d.page_content for d in docs)

#         # ------- RAG chain -------
#         self.rag_chain = (
#             {
#                 "context": self.retriever | RunnableLambda(format_docs),
#                 "question": RunnablePassthrough(),
#             }
#             | prompt
#             | self.chat_llm
#             | StrOutputParser()
#         )


# if __name__ == "__main__":
#     bot = ChatBot()
#     user_q = input("Ask me anything: ")
#     result = bot.rag_chain.invoke(user_q)
#     print(result)


# # main.py
# import os
# from typing import List, Optional, Literal, TypedDict

# from dotenv import load_dotenv

# # Pinecone + Vector store
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# # Embeddings / LLM
# from langchain_huggingface import (
#     HuggingFaceEmbeddings,
#     ChatHuggingFace,
#     HuggingFaceEndpoint,
# )

# # Data loading / splitting
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# # LangChain core
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda

# # Advanced retrieval
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain.retrievers import ContextualCompressionRetriever

# # Tools
# from langchain_core.tools import tool
# import hashlib, datetime

# # LangGraph
# from langgraph.graph import StateGraph, START, END

# # Memory (version-agnostic import)
# try:
#     # Newer LangChain
#     from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
# except ImportError:
#     # Older LangChain
#     from langchain_community.chat_message_histories import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory


# class ChatBot:
#     def __init__(
#         self,
#         index_name: str = "langchain-demo",
#         cloud: str = "aws",
#         region: str = "us-east-1",
#     ):
#         load_dotenv()

#         # HF env compatibility
#         if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None and os.getenv("HUGGINGFACEHUB_API_KEY"):
#             os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_KEY")

#         pinecone_key = os.getenv("PINECONE_API_KEY")
#         hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         if not pinecone_key or not hf_token:
#             raise RuntimeError("Missing PINECONE_API_KEY or HUGGINGFACEHUB_API_TOKEN in environment.")

#         # ------- Load & split docs -------
#         if not os.path.exists("./horoscope.txt"):
#             raise FileNotFoundError("Couldn't find './horoscope.txt'. Make sure the file exists.")
#         loader = TextLoader("./horoscope.txt")
#         documents = loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
#         docs = text_splitter.split_documents(documents)

#         # ------- Embeddings -------
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         dim = len(self.embeddings.embed_query("ping"))  # expected 768

#         # ------- Pinecone setup -------
#         pc = Pinecone(api_key=pinecone_key)
#         self.index_name = index_name

#         def _idx_name(x):
#             return x.name if hasattr(x, "name") else (x.get("name") if isinstance(x, dict) else None)

#         existing = {_idx_name(i) for i in pc.list_indexes()}
#         if self.index_name not in existing:
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=dim,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud=cloud, region=region),
#             )

#         idx = pc.Index(self.index_name)
#         stats = idx.describe_index_stats()
#         namespaces = stats.get("namespaces", {}) or {}
#         total = sum(ns.get("vector_count", 0) for ns in namespaces.values()) if namespaces else stats.get("total_vector_count", 0) or 0

#         if total == 0:
#             self.vectorstore = PineconeVectorStore.from_documents(docs, embedding=self.embeddings, index_name=self.index_name)
#         else:
#             self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

#         # ------- LLMs -------
#         HF_TOKEN = hf_token

#         # Final answerer (fluent)
#         self.chat_llm = ChatHuggingFace(
#             llm=HuggingFaceEndpoint(
#                 repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
#                 task="text-generation",
#                 temperature=0.7,
#                 do_sample=True,
#                 top_k=50,
#                 max_new_tokens=256,
#                 return_full_text=False,
#                 huggingfacehub_api_token=HF_TOKEN,
#             )
#         )

#         # Deterministic rewriter (can be same model; different decoding)
#         self.rewriter_llm = ChatHuggingFace(
#             llm=HuggingFaceEndpoint(
#                 repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
#                 task="text-generation",
#                 temperature=0.2,
#                 do_sample=False,
#                 max_new_tokens=96,
#                 return_full_text=False,
#                 huggingfacehub_api_token=HF_TOKEN,
#             )
#         )

#         # ------- Advanced retriever (MultiQuery + Cross-Encoder rerank + Compression) -------
#         base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
#         mqr = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.rewriter_llm)

#         reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")  # switch to -large if you want
#         cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model_name)
#         reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

#         self.advanced_retriever = ContextualCompressionRetriever(
#             base_compressor=reranker,
#             base_retriever=mqr,
#         )

#         # ------- Helpers -------
#         self.SIGNS = [
#             "aries","taurus","gemini","cancer","leo","virgo",
#             "libra","scorpio","sagittarius","capricorn","aquarius","pisces"
#         ]

#         def extract_sign_from_text(text: str) -> Optional[str]:
#             t = (text or "").lower()
#             for s in self.SIGNS:
#                 if s in t:
#                     return s
#             return None

#         def filter_docs_by_sign(docs_list: List[Document], sign: Optional[str]) -> List[Document]:
#             if not sign:
#                 return docs_list
#             s = sign.lower()
#             return [d for d in docs_list if s in (d.page_content or "").lower()]

#         def extract_citations(docs_list: List[Document]) -> List[str]:
#             cites = []
#             for i, d in enumerate(docs_list, 1):
#                 md = (getattr(d, "metadata", {}) or {})
#                 src = md.get("source") or md.get("file") or "horoscope.txt"
#                 cites.append(f"[{i}] {src}")
#             return cites

#         def format_context(docs_list: List[Document]) -> str:
#             return "\n\n".join(
#                 getattr(d, "page_content", "") for d in docs_list if getattr(d, "page_content", "")
#             )

#         # ------- Tools -------
#         @tool("lucky_number")
#         def lucky_number(name_or_sign: str) -> str:
#             """Deterministic 'lucky number' (1-9) from a name or zodiac sign."""
#             h = int(hashlib.md5(name_or_sign.strip().lower().encode("utf-8")).hexdigest(), 16)
#             num = (h % 9) + 1
#             return f"Lucky number for '{name_or_sign}': {num}"

#         @tool("now")
#         def now(_: str = "") -> str:
#             """Current date/time (ISO format)."""
#             return datetime.datetime.now().isoformat(timespec="seconds")

#         self.lucky_number = lucky_number
#         self.now = now

#         # ------- LangGraph (sign-aware): route → tools/RAG → grade → fallback/generate -------
#         class RAGState(TypedDict, total=False):
#             session_id: str
#             question: str
#             history: List
#             route: Literal["TOOLS", "RAG"]
#             target_sign: Optional[str]
#             docs: List[Document]
#             citations: List[str]
#             tool_result: Optional[str]
#             grounded: bool
#             answer: str

#         TOOL_KEYWORDS = ("lucky number", "lucky", "today", "date", "time", "now")

#         def route_node(state: RAGState) -> RAGState:
#             q = (state.get("question") or "")
#             route = "TOOLS" if any(k in q.lower() for k in TOOL_KEYWORDS) else "RAG"
#             return {**state, "route": route, "target_sign": extract_sign_from_text(q)}

#         def tools_node(state: RAGState) -> RAGState:
#             q = state["question"]
#             if "lucky" in q.lower():
#                 target = extract_sign_from_text(q) or q.strip()
#                 result = self.lucky_number.invoke(target.title())
#             else:
#                 result = self.now.invoke("")
#             return {**state, "tool_result": result, "answer": result, "docs": [], "citations": [], "grounded": True}

#         def retrieve_node(state: RAGState) -> RAGState:
#             docs_list = self.advanced_retriever.invoke(state["question"])  # new API
#             sign = state.get("target_sign")
#             if sign:
#                 docs_list = filter_docs_by_sign(docs_list, sign)
#             return {**state, "docs": docs_list, "citations": extract_citations(docs_list)}

#         def grade_node(state: RAGState) -> RAGState:
#             grounded = len(state.get("docs") or []) > 0
#             return {**state, "grounded": grounded}

#         def fallback_node(state: RAGState) -> RAGState:
#             sign = state.get("target_sign")
#             if sign:
#                 msg = f"I don't have content for {sign.title()} yet. Please add it to the corpus."
#             else:
#                 msg = "I don't know yet. Please add relevant content to the corpus."
#             return {**state, "answer": msg}

#         rag_prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a helpful, concise fortune teller. Use ONLY the provided context and tools. "
#              "If you don't know, say you don't know. Keep answers within 2 short sentences."),
#             MessagesPlaceholder("history"),
#             ("human",
#              "Context:\n{context}\n\nTools:\n{tool_result}\n\nQuestion: {question}\n"
#              "Cite sources as [1], [2] if used.")
#         ])
#         generator = rag_prompt | self.chat_llm | StrOutputParser()

#         def generate_node(state: RAGState) -> RAGState:
#             # If tools already set the answer, skip generation
#             if state.get("tool_result"):
#                 return state
#             context = format_context(state.get("docs", [])) if state.get("docs") else ""
#             tool_text = state.get("tool_result", "")
#             answer = generator.invoke({
#                 "history": state.get("history", []),
#                 "context": context,
#                 "tool_result": tool_text,
#                 "question": state["question"],
#             })
#             return {**state, "answer": answer}

#         graph = StateGraph(RAGState)
#         graph.add_node("route", route_node)
#         graph.add_node("tools", tools_node)
#         graph.add_node("retrieve", retrieve_node)
#         graph.add_node("grade", grade_node)
#         graph.add_node("fallback", fallback_node)
#         graph.add_node("generate", generate_node)

#         graph.add_edge(START, "route")
#         graph.add_conditional_edges("route", lambda s: "tools" if s["route"] == "TOOLS" else "retrieve")
#         graph.add_edge("tools", END)  # tools short-circuit to END
#         graph.add_edge("retrieve", "grade")
#         graph.add_conditional_edges("grade", lambda s: "generate" if s["grounded"] else "fallback")
#         graph.add_edge("generate", END)
#         graph.add_edge("fallback", END)

#         self.graph_app = graph.compile()

#         # ------- Memory wrapper -------
#         self._session_store = {}

#         def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#             if session_id not in self._session_store:
#                 self._session_store[session_id] = InMemoryChatMessageHistory()
#             return self._session_store[session_id]

#         # Return only final string answer from the graph
#         answer_only = self.graph_app | RunnableLambda(lambda state: state["answer"])

#         self.graph_with_memory = RunnableWithMessageHistory(
#             answer_only,
#             get_session_history,
#             input_messages_key="question",
#             history_messages_key="history",
#         )

#         # ------- Back-compat: rag_chain.invoke(question) returns string via graph -------
#         class _Invoker:
#             def __init__(self, outer):
#                 self._outer = outer
#             def invoke(self, question: str, session_id: str = "default"):
#                 cfg = {"configurable": {"session_id": session_id}}
#                 return self._outer.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)

#         self.rag_chain = _Invoker(self)

#     # Convenience method for callers
#     def answer_with_graph(self, question: str, session_id: str = "default") -> str:
#         cfg = {"configurable": {"session_id": session_id}}
#         return self.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)


# if __name__ == "__main__":
#     bot = ChatBot()
#     try:
#         q = input("Ask me anything: ")
#     except EOFError:
#         q = "What can Sagittarius expect this week?"
#     ans = bot.answer_with_graph(q, session_id="cli-user")
#     print(ans)

# # main.py
# import os
# import re
# from typing import List, Optional, Literal, TypedDict
# from collections import Counter

# from dotenv import load_dotenv

# # Pinecone + Vector store
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# # Embeddings (local HF download; NOT HF inference)
# from langchain_huggingface import HuggingFaceEmbeddings

# # Data loading / splitting
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# # LangChain core
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda

# # Advanced retrieval
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain.retrievers import ContextualCompressionRetriever

# # Tools
# from langchain_core.tools import tool
# import hashlib, datetime

# # LangGraph
# from langgraph.graph import StateGraph, START, END

# # Memory (version-agnostic)
# try:
#     from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
# except ImportError:
#     from langchain_community.chat_message_histories import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # LLMs via Groq
# from langchain_groq import ChatGroq


# class ChatBot:
#     def __init__(
#         self,
#         index_name: str = "langchain-demo-signed-v2",
#         cloud: str = "aws",
#         region: str = "us-east-1",
#     ):
#         load_dotenv()

#         # --- Required env ---
#         pinecone_key = os.getenv("PINECONE_API_KEY")
#         groq_key = os.getenv("GROQ_API_KEY")
#         if not pinecone_key:
#             raise RuntimeError("Missing PINECONE_API_KEY in environment.")
#         if not groq_key:
#             raise RuntimeError("Missing GROQ_API_KEY in environment.")

#         # --- Load base docs ---
#         if not os.path.exists("./horoscope.txt"):
#             raise FileNotFoundError("Couldn't find './horoscope.txt'. Make sure the file exists.")
#         loader = TextLoader("./horoscope.txt")
#         base_docs = loader.load()  # list[Document]

#         # --- Tag each BASE document once with its dominant sign (no nulls) ---
#         SIGN_LIST = [
#             "aries","taurus","gemini","cancer","leo","virgo",
#             "libra","scorpio","sagittarius","capricorn","aquarius","pisces"
#         ]
#         sign_re = re.compile(r"\b(" + "|".join(SIGN_LIST) + r")\b", flags=re.I)

#         for d in base_docs:
#             text = d.page_content or ""
#             found = [s.lower() for s in sign_re.findall(text)]
#             d.metadata = (d.metadata or {})
#             if found:
#                 dominant = Counter(found).most_common(1)[0][0]
#                 d.metadata["sign"] = dominant      # e.g., "sagittarius"
#             else:
#                 d.metadata.pop("sign", None)       # ensure no null/None
#             d.metadata.setdefault("source", "horoscope.txt")

#         # --- Split AFTER tagging so chunks inherit metadata['sign'] ---
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
#         docs = text_splitter.split_documents(base_docs)

#         # --- Embeddings ---
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         dim = len(self.embeddings.embed_query("ping"))  # expected 768

#         # --- Pinecone setup ---
#         pc = Pinecone(api_key=pinecone_key)
#         self.index_name = index_name

#         def _idx_name(x):
#             return x.name if hasattr(x, "name") else (x.get("name") if isinstance(x, dict) else None)

#         existing = {_idx_name(i) for i in pc.list_indexes()}
#         if self.index_name not in existing:
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=dim,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud=cloud, region=region),
#             )

#         idx = pc.Index(self.index_name)
#         stats = idx.describe_index_stats()
#         namespaces = stats.get("namespaces", {}) or {}
#         total = sum(ns.get("vector_count", 0) for ns in namespaces.values()) if namespaces else stats.get("total_vector_count", 0) or 0

#         # If index is empty, upsert; otherwise connect
#         if total == 0:
#             self.vectorstore = PineconeVectorStore.from_documents(
#                 docs, embedding=self.embeddings, index_name=self.index_name
#             )
#         else:
#             self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

#         # --- LLMs (Groq) ---
#         self.chat_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=200)
#         self.rewriter_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=96)

#         # --- Advanced retrieval (MultiQuery + rerank + compression) ---
#         base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
#         mqr = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.rewriter_llm)

#         reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
#         cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model_name)
#         reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

#         self.advanced_retriever = ContextualCompressionRetriever(
#             base_compressor=reranker,
#             base_retriever=mqr,
#         )

#         # --- Helpers ---
#         self.SIGNS = SIGN_LIST

#         def extract_sign_from_text(text: str) -> Optional[str]:
#             t = (text or "").lower()
#             for s in self.SIGNS:
#                 if s in t:
#                     return s
#             return None

#         # STRICT: keep only docs whose metadata["sign"] matches the target sign
#         def filter_docs_by_sign(docs_list: List[Document], sign: Optional[str]) -> List[Document]:
#             if not sign:
#                 return docs_list
#             s = sign.lower()
#             keep: List[Document] = []
#             for d in docs_list:
#                 md_sign = ((d.metadata or {}).get("sign") or "").lower()
#                 if md_sign == s:
#                     keep.append(d)
#             return keep

#         def extract_citations(docs_list: List[Document]) -> List[str]:
#             cites = []
#             for i, d in enumerate(docs_list, 1):
#                 md = (getattr(d, "metadata", {}) or {})
#                 src = md.get("source") or md.get("file") or "horoscope.txt"
#                 cites.append(f"[{i}] {src}")
#             return cites

#         def format_context(docs_list: List[Document]) -> str:
#             return "\n\n".join(
#                 getattr(d, "page_content", "") for d in docs_list if getattr(d, "page_content", "")
#             )

#         def is_general_question(q: str) -> bool:
#             """Heuristic: True for general knowledge/how-to questions (not astrology/tools)."""
#             ql = (q or "").lower()
#             if any(s in ql for s in self.SIGNS):
#                 return False
#             if any(k in ql for k in ("lucky number", "lucky", "today", "date", "time", "now")):
#                 return False
#             GENERAL_HINTS = ("capital", "recipe", "how to", "who is", "what is", "define",
#                              "population", "country", "city", "explain", "difference between")
#             return any(h in ql for h in GENERAL_HINTS)

#         self.extract_sign_from_text = extract_sign_from_text
#         self.filter_docs_by_sign = filter_docs_by_sign
#         self.extract_citations = extract_citations
#         self.format_context = format_context
#         self.is_general_question = is_general_question

#         # --- Tools ---
#         @tool("lucky_number")
#         def lucky_number(name_or_sign: str) -> str:
#             """Deterministic 'lucky number' (1-9) from a name or zodiac sign."""
#             h = int(hashlib.md5(name_or_sign.strip().lower().encode("utf-8")).hexdigest(), 16)
#             num = (h % 9) + 1
#             return f"Lucky number for '{name_or_sign}': {num}"

#         @tool("now")
#         def now(_: str = "") -> str:
#             """Current date/time (ISO format)."""
#             return datetime.datetime.now().isoformat(timespec="seconds")

#         self.lucky_number = lucky_number
#         self.now = now

#         # --- LangGraph: route → tools/GENERAL/RAG → grade → fallback/generate ---
#         class RAGState(TypedDict, total=False):
#             session_id: str
#             question: str
#             history: List
#             route: Literal["TOOLS", "GENERAL", "RAG"]
#             target_sign: Optional[str]
#             docs: List[Document]
#             citations: List[str]
#             tool_result: Optional[str]
#             grounded: bool
#             answer: str

#         TOOL_KEYWORDS = ("lucky number", "lucky", "today", "date", "time", "now")

#         # --- Prompts ---
#         rag_prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a concise fortune teller.\n"
#              "- Only answer about the user's sign: {target_sign} (if provided).\n"
#              "- Use ONLY the provided context and tool output.\n"
#              "- Answer the user's single question in 1–2 sentences.\n"
#              "- Do NOT invent new questions or headings.\n"
#              "- If the context is empty or about a different sign, reply exactly: I don't know."),
#             MessagesPlaceholder("history"),
#             ("human",
#              "Context:\n{context}\n\nTool:\n{tool_result}\n\nUser question: {question}\n"
#              "Citations (optional): {citations}")
#         ])
#         generator = rag_prompt | self.chat_llm | StrOutputParser()

#         general_prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a helpful assistant. Answer the user clearly and concisely in 2–3 sentences."),
#             MessagesPlaceholder("history"),
#             ("human", "{question}")
#         ])
#         general_generator = general_prompt | self.chat_llm | StrOutputParser()

#         # --- Nodes ---
#         def route_node(state: RAGState) -> RAGState:
#             q = (state.get("question") or "")
#             if any(k in q.lower() for k in TOOL_KEYWORDS):
#                 route = "TOOLS"
#             elif self.is_general_question(q):
#                 route = "GENERAL"
#             else:
#                 route = "RAG"
#             return {**state, "route": route, "target_sign": self.extract_sign_from_text(q)}

#         def tools_node(state: RAGState) -> RAGState:
#             q = state["question"]
#             if "lucky" in q.lower():
#                 target = self.extract_sign_from_text(q) or q.strip()
#                 result = self.lucky_number.invoke(target.title())
#             else:
#                 result = self.now.invoke("")
#             return {**state, "tool_result": result, "answer": result, "docs": [], "citations": [], "grounded": True}

#         def general_node(state: RAGState) -> RAGState:
#             ans = general_generator.invoke({
#                 "history": state.get("history", []),
#                 "question": state["question"]
#             })
#             return {**state, "answer": ans, "docs": [], "citations": [], "grounded": False}

#         def retrieve_node(state: RAGState) -> RAGState:
#             q = state["question"]
#             sign = state.get("target_sign")
#             biased_query = f"[{sign}] {q}" if sign else q
#             docs_list = self.advanced_retriever.invoke(biased_query)
#             if sign:
#                 docs_list = self.filter_docs_by_sign(docs_list, sign)
#             return {**state, "docs": docs_list, "citations": self.extract_citations(docs_list)}

#         def grade_node(state: RAGState) -> RAGState:
#             grounded = len(state.get("docs") or []) > 0
#             return {**state, "grounded": grounded}

#         def fallback_node(state: RAGState) -> RAGState:
#             # Non-general, horoscope-style query with no matching context
#             sign = state.get("target_sign")
#             if sign:
#                 msg = f"I don't have content for {sign.title()} yet. Please add it to the corpus."
#             else:
#                 msg = "I don't know yet. Please add relevant content to the corpus."
#             return {**state, "answer": msg, "grounded": False}

#         def generate_node(state: RAGState) -> RAGState:
#             if state.get("tool_result"):
#                 return state
#             if not state.get("docs"):
#                 return fallback_node(state)

#             context = self.format_context(state.get("docs", []))
#             tool_text = state.get("tool_result", "")
#             cites = " ".join(state.get("citations", []))
#             answer = generator.invoke({
#                 "history": state.get("history", []),
#                 "context": context,
#                 "tool_result": tool_text,
#                 "question": state["question"],
#                 "citations": cites,
#                 "target_sign": (state.get("target_sign") or ""),
#             })
#             answer = re.sub(r'(?mi)^\s*Question:.*$', '', answer).strip()
#             return {**state, "answer": answer}

#         # --- Graph wiring ---
#         graph = StateGraph(RAGState)
#         graph.add_node("route", route_node)
#         graph.add_node("tools", tools_node)
#         graph.add_node("general", general_node)
#         graph.add_node("retrieve", retrieve_node)
#         graph.add_node("grade", grade_node)
#         graph.add_node("fallback", fallback_node)
#         graph.add_node("generate", generate_node)

#         def _router(s: RAGState) -> str:
#             return "tools" if s["route"] == "TOOLS" else ("general" if s["route"] == "GENERAL" else "retrieve")

#         graph.add_edge(START, "route")
#         graph.add_conditional_edges("route", _router)
#         graph.add_edge("tools", END)
#         graph.add_edge("general", END)
#         graph.add_edge("retrieve", "grade")
#         graph.add_conditional_edges("grade", lambda s: "generate" if s["grounded"] else "fallback")
#         graph.add_edge("generate", END)
#         graph.add_edge("fallback", END)

#         self.graph_app = graph.compile()

#         # --- Memory wrapper (ephemeral, in-process) ---
#         self._session_store = {}
#         def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#             if session_id not in self._session_store:
#                 self._session_store[session_id] = InMemoryChatMessageHistory()
#             return self._session_store[session_id]

#         answer_only = self.graph_app | RunnableLambda(lambda state: state["answer"])
#         self.graph_with_memory = RunnableWithMessageHistory(
#             answer_only,
#             get_session_history,
#             input_messages_key="question",
#             history_messages_key="history",
#         )

#         # Back-compat for Streamlit
#         class _Invoker:
#             def __init__(self, outer):
#                 self._outer = outer
#             def invoke(self, question: str, session_id: str = "default"):
#                 cfg = {"configurable": {"session_id": session_id}}
#                 return self._outer.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)
#         self.rag_chain = _Invoker(self)

#     # Convenience method
#     def answer_with_graph(self, question: str, session_id: str = "default") -> str:
#         cfg = {"configurable": {"session_id": session_id}}
#         return self.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)


# if __name__ == "__main__":
#     bot = ChatBot()
#     try:
#         q = input("Ask me anything: ")
#     except EOFError:
#         q = "What can Sagittarius expect this week?"
#     ans = bot.answer_with_graph(q, session_id="cli-user")
#     print(ans)

# main.py
# Added a tiny “memory API” to your ChatBot so you can list sessions, inspect a session’s messages, clear them, and save/load to JSON
# import os
# import re
# import json
# from typing import List, Optional, Literal, TypedDict
# from collections import Counter

# from dotenv import load_dotenv

# # Pinecone + Vector store
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# # Embeddings (local HF download; NOT HF inference)
# from langchain_huggingface import HuggingFaceEmbeddings

# # Data loading / splitting
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# # LangChain core
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda
# from langchain_core.messages import messages_to_dict, messages_from_dict  # <-- for memory export/import

# # Advanced retrieval
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain.retrievers import ContextualCompressionRetriever

# # Tools
# from langchain_core.tools import tool
# import hashlib, datetime

# # LangGraph
# from langgraph.graph import StateGraph, START, END

# # Memory (version-agnostic)
# try:
#     from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
# except ImportError:
#     from langchain_community.chat_message_histories import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # LLMs via Groq
# from langchain_groq import ChatGroq


# class ChatBot:
#     def __init__(
#         self,
#         index_name: str = "langchain-demo-signed-v2",  # NEW name to force clean reindex
#         cloud: str = "aws",
#         region: str = "us-east-1",
#     ):
#         load_dotenv()

#         # --- Required env ---
#         pinecone_key = os.getenv("PINECONE_API_KEY")
#         groq_key = os.getenv("GROQ_API_KEY")
#         if not pinecone_key:
#             raise RuntimeError("Missing PINECONE_API_KEY in environment.")
#         if not groq_key:
#             raise RuntimeError("Missing GROQ_API_KEY in environment.")

#         # --- Load base docs ---
#         if not os.path.exists("./horoscope.txt"):
#             raise FileNotFoundError("Couldn't find './horoscope.txt'. Make sure the file exists.")
#         loader = TextLoader("./horoscope.txt")
#         base_docs = loader.load()  # list[Document]

#         # --- Tag each BASE document once with its dominant sign (no nulls) ---
#         SIGN_LIST = [
#             "aries","taurus","gemini","cancer","leo","virgo",
#             "libra","scorpio","sagittarius","capricorn","aquarius","pisces"
#         ]
#         sign_re = re.compile(r"\b(" + "|".join(SIGN_LIST) + r")\b", flags=re.I)

#         for d in base_docs:
#             text = d.page_content or ""
#             found = [s.lower() for s in sign_re.findall(text)]
#             d.metadata = (d.metadata or {})
#             if found:
#                 dominant = Counter(found).most_common(1)[0][0]
#                 d.metadata["sign"] = dominant      # e.g., "sagittarius"
#             else:
#                 d.metadata.pop("sign", None)       # ensure no null/None
#             d.metadata.setdefault("source", "horoscope.txt")

#         # --- Split AFTER tagging so chunks inherit metadata['sign'] ---
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
#         docs = text_splitter.split_documents(base_docs)

#         # --- Embeddings ---
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         dim = len(self.embeddings.embed_query("ping"))  # expected 768

#         # --- Pinecone setup ---
#         pc = Pinecone(api_key=pinecone_key)
#         self.index_name = index_name

#         def _idx_name(x):
#             return x.name if hasattr(x, "name") else (x.get("name") if isinstance(x, dict) else None)

#         existing = {_idx_name(i) for i in pc.list_indexes()}
#         if self.index_name not in existing:
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=dim,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud=cloud, region=region),
#             )

#         idx = pc.Index(self.index_name)
#         stats = idx.describe_index_stats()
#         namespaces = stats.get("namespaces", {}) or {}
#         total = sum(ns.get("vector_count", 0) for ns in namespaces.values()) if namespaces else stats.get("total_vector_count", 0) or 0

#         # If index is empty, upsert; otherwise connect
#         if total == 0:
#             self.vectorstore = PineconeVectorStore.from_documents(
#                 docs, embedding=self.embeddings, index_name=self.index_name
#             )
#         else:
#             self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

#         # --- LLMs (Groq) ---
#         self.chat_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=200)
#         self.rewriter_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=96)

#         # --- Advanced retrieval (MultiQuery + rerank + compression) ---
#         base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
#         mqr = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.rewriter_llm)

#         reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
#         cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model_name)
#         reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

#         self.advanced_retriever = ContextualCompressionRetriever(
#             base_compressor=reranker,
#             base_retriever=mqr,
#         )

#         # --- Helpers ---
#         self.SIGNS = SIGN_LIST

#         def extract_sign_from_text(text: str) -> Optional[str]:
#             t = (text or "").lower()
#             for s in self.SIGNS:
#                 if s in t:
#                     return s
#             return None

#         # STRICT: keep only docs whose metadata["sign"] matches the target sign
#         def filter_docs_by_sign(docs_list: List[Document], sign: Optional[str]) -> List[Document]:
#             if not sign:
#                 return docs_list
#             s = sign.lower()
#             keep: List[Document] = []
#             for d in docs_list:
#                 md_sign = ((d.metadata or {}).get("sign") or "").lower()
#                 if md_sign == s:
#                     keep.append(d)
#             return keep

#         def extract_citations(docs_list: List[Document]) -> List[str]:
#             cites = []
#             for i, d in enumerate(docs_list, 1):
#                 md = (getattr(d, "metadata", {}) or {})
#                 src = md.get("source") or md.get("file") or "horoscope.txt"
#                 cites.append(f"[{i}] {src}")
#             return cites

#         def format_context(docs_list: List[Document]) -> str:
#             return "\n\n".join(
#                 getattr(d, "page_content", "") for d in docs_list if getattr(d, "page_content", "")
#             )

#         def is_general_question(q: str) -> bool:
#             """Heuristic: True for general knowledge/how-to questions (not astrology/tools)."""
#             ql = (q or "").lower()
#             if any(s in ql for s in self.SIGNS):
#                 return False
#             if any(k in ql for k in ("lucky number", "lucky", "today", "date", "time", "now")):
#                 return False
#             GENERAL_HINTS = ("capital", "recipe", "how to", "who is", "what is", "define",
#                              "population", "country", "city", "explain", "difference between")
#             return any(h in ql for h in GENERAL_HINTS)

#         self.extract_sign_from_text = extract_sign_from_text
#         self.filter_docs_by_sign = filter_docs_by_sign
#         self.extract_citations = extract_citations
#         self.format_context = format_context
#         self.is_general_question = is_general_question

#         # --- Tools ---
#         @tool("lucky_number")
#         def lucky_number(name_or_sign: str) -> str:
#             """Deterministic 'lucky number' (1-9) from a name or zodiac sign."""
#             h = int(hashlib.md5(name_or_sign.strip().lower().encode("utf-8")).hexdigest(), 16)
#             num = (h % 9) + 1
#             return f"Lucky number for '{name_or_sign}': {num}"

#         @tool("now")
#         def now(_: str = "") -> str:
#             """Current date/time (ISO format)."""
#             return datetime.datetime.now().isoformat(timespec="seconds")

#         self.lucky_number = lucky_number
#         self.now = now

#         # --- LangGraph: route → tools/GENERAL/RAG → grade → fallback/generate ---
#         class RAGState(TypedDict, total=False):
#             session_id: str
#             question: str
#             history: List
#             route: Literal["TOOLS", "GENERAL", "RAG"]
#             target_sign: Optional[str]
#             docs: List[Document]
#             citations: List[str]
#             tool_result: Optional[str]
#             grounded: bool
#             answer: str

#         TOOL_KEYWORDS = ("lucky number", "lucky", "today", "date", "time", "now")

#         # --- Prompts ---
#         rag_prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a concise fortune teller.\n"
#              "- Only answer about the user's sign: {target_sign} (if provided).\n"
#              "- Use ONLY the provided context and tool output.\n"
#              "- Answer the user's single question in 1–2 sentences.\n"
#              "- Do NOT invent new questions or headings.\n"
#              "- If the context is empty or about a different sign, reply exactly: I don't know."),
#             MessagesPlaceholder("history"),
#             ("human",
#              "Context:\n{context}\n\nTool:\n{tool_result}\n\nUser question: {question}\n"
#              "Citations (optional): {citations}")
#         ])
#         generator = rag_prompt | self.chat_llm | StrOutputParser()

#         general_prompt = ChatPromptTemplate.from_messages([
#             ("system",
#              "You are a helpful assistant. Answer the user clearly and concisely in 2–3 sentences."),
#             MessagesPlaceholder("history"),
#             ("human", "{question}")
#         ])
#         general_generator = general_prompt | self.chat_llm | StrOutputParser()

#         # --- Nodes ---
#         def route_node(state: RAGState) -> RAGState:
#             q = (state.get("question") or "")
#             if any(k in q.lower() for k in TOOL_KEYWORDS):
#                 route = "TOOLS"
#             elif self.is_general_question(q):
#                 route = "GENERAL"
#             else:
#                 route = "RAG"
#             return {**state, "route": route, "target_sign": self.extract_sign_from_text(q)}

#         def tools_node(state: RAGState) -> RAGState:
#             q = state["question"]
#             if "lucky" in q.lower():
#                 target = self.extract_sign_from_text(q) or q.strip()
#                 result = self.lucky_number.invoke(target.title())
#             else:
#                 result = self.now.invoke("")
#             return {**state, "tool_result": result, "answer": result, "docs": [], "citations": [], "grounded": True}

#         def general_node(state: RAGState) -> RAGState:
#             ans = general_generator.invoke({
#                 "history": state.get("history", []),
#                 "question": state["question"]
#             })
#             return {**state, "answer": ans, "docs": [], "citations": [], "grounded": False}

#         def retrieve_node(state: RAGState) -> RAGState:
#             q = state["question"]
#             sign = state.get("target_sign")
#             biased_query = f"[{sign}] {q}" if sign else q
#             docs_list = self.advanced_retriever.invoke(biased_query)
#             if sign:
#                 docs_list = self.filter_docs_by_sign(docs_list, sign)
#             return {**state, "docs": docs_list, "citations": self.extract_citations(docs_list)}

#         def grade_node(state: RAGState) -> RAGState:
#             grounded = len(state.get("docs") or []) > 0
#             return {**state, "grounded": grounded}

#         def fallback_node(state: RAGState) -> RAGState:
#             # Non-general, horoscope-style query with no matching context
#             sign = state.get("target_sign")
#             if sign:
#                 msg = f"I don't have content for {sign.title()} yet. Please add it to the corpus."
#             else:
#                 msg = "I don't know yet. Please add relevant content to the corpus."
#             return {**state, "answer": msg, "grounded": False}

#         def generate_node(state: RAGState) -> RAGState:
#             if state.get("tool_result"):
#                 return state
#             if not state.get("docs"):
#                 return fallback_node(state)

#             context = self.format_context(state.get("docs", []))
#             tool_text = state.get("tool_result", "")
#             cites = " ".join(state.get("citations", []))
#             answer = generator.invoke({
#                 "history": state.get("history", []),
#                 "context": context,
#                 "tool_result": tool_text,
#                 "question": state["question"],
#                 "citations": cites,
#                 "target_sign": (state.get("target_sign") or ""),
#             })
#             answer = re.sub(r'(?mi)^\s*Question:.*$', '', answer).strip()
#             return {**state, "answer": answer}

#         # --- Graph wiring ---
#         graph = StateGraph(RAGState)
#         graph.add_node("route", route_node)
#         graph.add_node("tools", tools_node)
#         graph.add_node("general", general_node)
#         graph.add_node("retrieve", retrieve_node)
#         graph.add_node("grade", grade_node)
#         graph.add_node("fallback", fallback_node)
#         graph.add_node("generate", generate_node)

#         def _router(s: RAGState) -> str:
#             return "tools" if s["route"] == "TOOLS" else ("general" if s["route"] == "GENERAL" else "retrieve")

#         graph.add_edge(START, "route")
#         graph.add_conditional_edges("route", _router)
#         graph.add_edge("tools", END)
#         graph.add_edge("general", END)
#         graph.add_edge("retrieve", "grade")
#         graph.add_conditional_edges("grade", lambda s: "generate" if s["grounded"] else "fallback")
#         graph.add_edge("generate", END)
#         graph.add_edge("fallback", END)

#         self.graph_app = graph.compile()

#         # --- Memory wrapper (ephemeral, in-process) ---
#         self._session_store = {}
#         def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#             if session_id not in self._session_store:
#                 self._session_store[session_id] = InMemoryChatMessageHistory()
#             return self._session_store[session_id]

#         answer_only = self.graph_app | RunnableLambda(lambda state: state["answer"])
#         self.graph_with_memory = RunnableWithMessageHistory(
#             answer_only,
#             get_session_history,
#             input_messages_key="question",
#             history_messages_key="history",
#         )

#         # Back-compat for Streamlit
#         class _Invoker:
#             def __init__(self, outer):
#                 self._outer = outer
#             def invoke(self, question: str, session_id: str = "default"):
#                 cfg = {"configurable": {"session_id": session_id}}
#                 return self._outer.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)
#         self.rag_chain = _Invoker(self)

#     # ---------------- Memory utilities (NEW) ----------------
#     def list_sessions(self) -> List[str]:
#         """Return all session_ids currently stored in memory."""
#         return list(self._session_store.keys())

#     def get_history(self, session_id: str) -> List[dict]:
#         """Return message history for a session as a list of dicts (safe to print/JSON)."""
#         hist = self._session_store.get(session_id)
#         return messages_to_dict(hist.messages) if hist else []

#     def clear_history(self, session_id: str) -> None:
#         """Delete memory for a session id."""
#         self._session_store.pop(session_id, None)

#     def save_history(self, session_id: str, path: str) -> None:
#         """Persist a session's memory to a JSON file."""
#         hist = self._session_store.get(session_id)
#         if not hist:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(messages_to_dict(hist.messages), f, indent=2, ensure_ascii=False)

#     def load_history(self, session_id: str, path: str) -> None:
#         """Load memory from a JSON file into the given session id (overwrites existing)."""
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         msgs = messages_from_dict(data)
#         h = InMemoryChatMessageHistory()
#         # add_messages not guaranteed in all versions; fall back to add_message
#         try:
#             h.add_messages(msgs)  # type: ignore[attr-defined]
#         except Exception:
#             for m in msgs:
#                 h.add_message(m)
#         self._session_store[session_id] = h

#     # Convenience method for app code
#     def answer_with_graph(self, question: str, session_id: str = "default") -> str:
#         cfg = {"configurable": {"session_id": session_id}}
#         return self.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)


# if __name__ == "__main__":
#     bot = ChatBot()
#     try:
#         q = input("Ask me anything: ")
#     except EOFError:
#         q = "What can Sagittarius expect this week?"
#     ans = bot.answer_with_graph(q, session_id="cli-user")
#     print(ans)

#     # --- demo the memory helpers in CLI ---
#     sid = "cli-user"
#     print("\nSessions:", bot.list_sessions())
#     print("History preview:", bot.get_history(sid))
#     bot.save_history(sid, "memory_cli-user.json")
#     print('Saved to memory_cli-user.json')


# main.py
import os
import re
import json
import sqlite3
from typing import List, Optional, Literal, TypedDict
from collections import Counter

from dotenv import load_dotenv

# Pinecone + Vector store
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Embeddings (local HF download; NOT HF inference)
from langchain_huggingface import HuggingFaceEmbeddings

# Data loading / splitting
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# LangChain core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import messages_to_dict, messages_from_dict

# Advanced retrieval
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# Tools
from langchain_core.tools import tool
import hashlib, datetime

# LangGraph
from langgraph.graph import StateGraph, START, END

# Persistent Memory (SQLite)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# SQLAlchemy engine (to avoid deprecation warning)
from sqlalchemy import create_engine

# LLMs via Groq
from langchain_groq import ChatGroq


class ChatBot:
    def __init__(
        self,
        index_name: str = "langchain-demo-signed-v2",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        load_dotenv()

        # --- Required env ---
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if not pinecone_key:
            raise RuntimeError("Missing PINECONE_API_KEY in environment.")
        if not groq_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment.")

        # --- SQLite memory config ---
        self._db_path = os.getenv("MEMORY_DB_PATH", "memory.db")
        self._conn_str = f"sqlite:///{os.path.abspath(self._db_path)}"
        self._table_name = os.getenv("MEMORY_TABLE", "message_store")
        os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
        # Create a SQLAlchemy engine (recommended by LangChain; avoids deprecation warnings)
        self._engine = create_engine(self._conn_str)

        # --- Load base docs ---
        if not os.path.exists("./horoscope.txt"):
            raise FileNotFoundError("Couldn't find './horoscope.txt'. Make sure the file exists.")
        loader = TextLoader("./horoscope.txt")
        base_docs = loader.load()  # list[Document]

        # --- Tag each BASE document once with its dominant sign (no nulls) ---
        SIGN_LIST = [
            "aries","taurus","gemini","cancer","leo","virgo",
            "libra","scorpio","sagittarius","capricorn","aquarius","pisces"
        ]
        self._sign_word_regex = re.compile(r"\b(" + "|".join(SIGN_LIST) + r")\b", flags=re.I)

        for d in base_docs:
            text = d.page_content or ""
            found = [s.lower() for s in self._sign_word_regex.findall(text)]
            d.metadata = (d.metadata or {})
            if found:
                dominant = Counter(found).most_common(1)[0][0]
                d.metadata["sign"] = dominant
            else:
                d.metadata.pop("sign", None)
            d.metadata.setdefault("source", "horoscope.txt")

        # --- Split AFTER tagging so chunks inherit metadata['sign'] ---
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(base_docs)

        # --- Embeddings ---
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        dim = len(self.embeddings.embed_query("ping"))

        # --- Pinecone setup ---
        pc = Pinecone(api_key=pinecone_key)
        self.index_name = index_name

        def _idx_name(x):
            return x.name if hasattr(x, "name") else (x.get("name") if isinstance(x, dict) else None)

        existing = {_idx_name(i) for i in pc.list_indexes()}
        if self.index_name not in existing:
            pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        idx = pc.Index(self.index_name)
        stats = idx.describe_index_stats()
        namespaces = stats.get("namespaces", {}) or {}
        total = sum(ns.get("vector_count", 0) for ns in namespaces.values()) if namespaces else stats.get("total_vector_count", 0) or 0

        if total == 0:
            self.vectorstore = PineconeVectorStore.from_documents(
                docs, embedding=self.embeddings, index_name=self.index_name
            )
        else:
            self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

        # --- LLMs (Groq) ---
        self.chat_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=200)
        self.rewriter_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=96)

        # --- Advanced retrieval (MultiQuery + rerank + compression) ---
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        mqr = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.rewriter_llm)

        reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model_name)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

        self.advanced_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=mqr,
        )

        # --- Routing helpers / keyword sets ---
        self.SIGNS = SIGN_LIST

        # Word-boundary tool regex (prevents 'now' matching inside 'know')
        self._tool_regex = re.compile(r"\b(lucky number|lucky|today|date|time|now)\b", re.I)
        self._lucky_regex = re.compile(r"\blucky\b", re.I)

        self.PROGRAMMING_HINTS = (
            "python", "code", "script", "snippet", "example", "function", "class",
            "streamlit", "langchain", "langgraph", "pinecone", "groq", "sqlite", "sql",
            "database", "api", "sdk", "model", "embedding", "retriever", "reranker",
            "prompt", "install", "setup", "configure", "docker", "render", "ec2", "aws"
        )
        self.GENERAL_HINTS = (
            "how to", "what is", "who is", "define", "explain",
            "difference between", "steps", "guide", "tutorial", "recipe", "capital"
        )
        self.CODE_REGEX = re.compile(r"```|(^|\s)(import |def |class )|pip install|conda install", re.I)

        def extract_sign_from_text(text: str) -> Optional[str]:
            if not text:
                return None
            m = self._sign_word_regex.search(text)
            return m.group(1).lower() if m else None

        def filter_docs_by_sign(docs_list: List[Document], sign: Optional[str]) -> List[Document]:
            if not sign:
                return docs_list
            s = sign.lower()
            keep: List[Document] = []
            for d in docs_list:
                md_sign = ((d.metadata or {}).get("sign") or "").lower()
                if md_sign == s:
                    keep.append(d)
            return keep

        def extract_citations(docs_list: List[Document]) -> List[str]:
            cites = []
            for i, d in enumerate(docs_list, 1):
                md = (getattr(d, "metadata", {}) or {})
                src = md.get("source") or md.get("file") or "horoscope.txt"
                cites.append(f"[{i}] {src}")
            return cites

        def format_context(docs_list: List[Document]) -> str:
            return "\n\n".join(
                getattr(d, "page_content", "") for d in docs_list if getattr(d, "page_content", "")
            )

        def is_general_question(q: str) -> bool:
            """
            Route general facts + programming/code/infra questions to GENERAL.
            Priority: tools -> programming/code -> sign -> general hints.
            """
            ql = (q or "").lower()

            # Tools handled elsewhere (word boundaries)
            if self._tool_regex.search(ql):
                return False

            # Programming/tech signals → GENERAL (checked BEFORE sign words)
            if any(k in ql for k in self.PROGRAMMING_HINTS) or self.CODE_REGEX.search(q or ""):
                return True

            # If it clearly talks about a sign, keep it out of GENERAL
            if self._sign_word_regex.search(ql):
                return False

            # Other broad fact-style hints → GENERAL
            return any(h in ql for h in self.GENERAL_HINTS)

        self.extract_sign_from_text = extract_sign_from_text
        self.filter_docs_by_sign = filter_docs_by_sign
        self.extract_citations = extract_citations
        self.format_context = format_context
        self.is_general_question = is_general_question

        # --- Tools ---
        @tool("lucky_number")
        def lucky_number(name_or_sign: str) -> str:
            """Deterministic 'lucky number' (1-9) from a name or zodiac sign."""
            h = int(hashlib.md5(name_or_sign.strip().lower().encode("utf-8")).hexdigest(), 16)
            num = (h % 9) + 1
            return f"Lucky number for '{name_or_sign}': {num}"

        @tool("now")
        def now(_: str = "") -> str:
            """Current date/time (ISO format)."""
            return datetime.datetime.now().isoformat(timespec="seconds")

        self.lucky_number = lucky_number
        self.now = now

        # --- LangGraph: route → tools/GENERAL/RAG → grade → fallback/generate ---
        class RAGState(TypedDict, total=False):
            session_id: str
            question: str
            history: List
            route: Literal["TOOLS", "GENERAL", "RAG"]
            target_sign: Optional[str]
            docs: List[Document]
            citations: List[str]
            tool_result: Optional[str]
            grounded: bool
            answer: str

        # --- Prompts ---
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a concise fortune teller.\n"
             "- Only answer about the user's sign: {target_sign} (if provided).\n"
             "- Use ONLY the provided context and tool output.\n"
             "- Answer the user's single question in 1–2 sentences.\n"
             "- Do NOT invent new questions or headings.\n"
             "- If the context is empty or about a different sign, reply exactly: I don't know."),
            MessagesPlaceholder("history"),
            ("human",
             "Context:\n{context}\n\nTool:\n{tool_result}\n\nUser question: {question}\n"
             "Citations (optional): {citations}")
        ])
        generator = rag_prompt | self.chat_llm | StrOutputParser()

        # Allow code output when asked
        general_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer clearly and directly. "
             "If the user requests code, provide correct, runnable Python code in a fenced markdown block."),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ])
        general_generator = general_prompt | self.chat_llm | StrOutputParser()

        # --- Nodes ---
        def route_node(state: RAGState) -> RAGState:
            q = (state.get("question") or "")
            ql = q.lower()
            if self._tool_regex.search(ql):
                route = "TOOLS"
            elif self.is_general_question(q):
                route = "GENERAL"
            else:
                route = "RAG"
            return {**state, "route": route, "target_sign": self.extract_sign_from_text(q)}

        def tools_node(state: RAGState) -> RAGState:
            q = state["question"]
            if self._lucky_regex.search(q):  # word-boundary lucky
                target = self.extract_sign_from_text(q) or q.strip()
                result = self.lucky_number.invoke(target.title())
            else:
                result = self.now.invoke("")
            return {**state, "tool_result": result, "answer": result, "docs": [], "citations": [], "grounded": True}

        def general_node(state: RAGState) -> RAGState:
            ans = general_generator.invoke({
                "history": state.get("history", []),
                "question": state["question"]
            })
            return {**state, "answer": ans, "docs": [], "citations": [], "grounded": False}

        def retrieve_node(state: RAGState) -> RAGState:
            q = state["question"]
            sign = state.get("target_sign")
            biased_query = f"[{sign}] {q}" if sign else q
            docs_list = self.advanced_retriever.invoke(biased_query)
            if sign:
                docs_list = self.filter_docs_by_sign(docs_list, sign)
            return {**state, "docs": docs_list, "citations": self.extract_citations(docs_list)}

        def grade_node(state: RAGState) -> RAGState:
            grounded = len(state.get("docs") or []) > 0
            return {**state, "grounded": grounded}

        def fallback_node(state: RAGState) -> RAGState:
            sign = state.get("target_sign")
            if sign:
                msg = f"I don't have content for {sign.title()} yet. Please add it to the corpus."
            else:
                msg = "I don't know yet. Please add relevant content to the corpus."
            return {**state, "answer": msg, "grounded": False}

        def generate_node(state: RAGState) -> RAGState:
            if state.get("tool_result"):
                return state
            if not state.get("docs"):
                return fallback_node(state)

            context = self.format_context(state.get("docs", []))
            tool_text = state.get("tool_result", "")
            cites = " ".join(state.get("citations", []))
            answer = generator.invoke({
                "history": state.get("history", []),
                "context": context,
                "tool_result": tool_text,
                "question": state["question"],
                "citations": cites,
                "target_sign": (state.get("target_sign") or ""),
            })
            answer = re.sub(r'(?mi)^\s*Question:.*$', '', answer).strip()
            return {**state, "answer": answer}

        # --- Graph wiring ---
        graph = StateGraph(RAGState)
        graph.add_node("route", route_node)
        graph.add_node("tools", tools_node)
        graph.add_node("general", general_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade", grade_node)
        graph.add_node("fallback", fallback_node)
        graph.add_node("generate", generate_node)

        def _router(s: RAGState) -> str:
            return "tools" if s["route"] == "TOOLS" else ("general" if s["route"] == "GENERAL" else "retrieve")

        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", _router)
        graph.add_edge("tools", END)
        graph.add_edge("general", END)
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", lambda s: "generate" if s["grounded"] else "fallback")
        graph.add_edge("generate", END)
        graph.add_edge("fallback", END)

        self.graph_app = graph.compile()

        # --- Persistent memory wrapper (SQLite via SQLAlchemy engine) ---
        def get_session_history(session_id: str) -> SQLChatMessageHistory:
            return SQLChatMessageHistory(
                session_id=session_id,
                connection=self._engine,           # <- use engine, not deprecated connection_string
                table_name=self._table_name,
            )

        answer_only = self.graph_app | RunnableLambda(lambda state: state["answer"])
        self.graph_with_memory = RunnableWithMessageHistory(
            answer_only,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Back-compat for Streamlit
        class _Invoker:
            def __init__(self, outer):
                self._outer = outer
            def invoke(self, question: str, session_id: str = "default"):
                cfg = {"configurable": {"session_id": session_id}}
                return self._outer.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)
        self.rag_chain = _Invoker(self)

    # ---------------- Memory utilities (SQLite) ----------------
    def list_sessions(self) -> List[str]:
        """List distinct session_ids stored in SQLite."""
        try:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (self._table_name,))
            if not cur.fetchone():
                return []
            cur.execute(f"SELECT DISTINCT session_id FROM {self._table_name};")
            rows = cur.fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            try:
                con.close()
            except Exception:
                pass

    def get_history(self, session_id: str) -> List[dict]:
        """Return message history for a session as list[dict]."""
        hist = SQLChatMessageHistory(
            session_id=session_id,
            connection=self._engine,
            table_name=self._table_name,
        )
        return messages_to_dict(hist.messages)

    def clear_history(self, session_id: str) -> None:
        """Delete all messages for a session id."""
        hist = SQLChatMessageHistory(
            session_id=session_id,
            connection=self._engine,
            table_name=self._table_name,
        )
        hist.clear()

    def save_history(self, session_id: str, path: str) -> None:
        """Export a session's history to a JSON file."""
        data = self.get_history(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_history(self, session_id: str, path: str) -> None:
        """Import messages from a JSON file into the given session id (overwrites existing)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = messages_from_dict(data)
        hist = SQLChatMessageHistory(
            session_id=session_id,
            connection=self._engine,
            table_name=self._table_name,
        )
        hist.clear()
        for m in msgs:
            hist.add_message(m)

    # Convenience method for app code
    def answer_with_graph(self, question: str, session_id: str = "default") -> str:
        cfg = {"configurable": {"session_id": session_id}}
        return self.graph_with_memory.invoke({"session_id": session_id, "question": question}, config=cfg)


if __name__ == "__main__":
    bot = ChatBot()
    try:
        q = input("Ask me anything: ")
    except EOFError:
        q = "What can Sagittarius expect this week?"
    ans = bot.answer_with_graph(q, session_id="cli-user")
    print(ans)

    # --- demo the memory helpers in CLI ---
    sid = "cli-user"
    print("\nSessions:", bot.list_sessions())
    print("History preview:", bot.get_history(sid))
    bot.save_history(sid, "memory_cli-user.json")
    print('Saved to memory_cli-user.json')
