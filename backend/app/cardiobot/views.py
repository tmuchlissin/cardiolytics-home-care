import os
import json
import numpy as np
import time
import re
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Dict, Any
from rapidfuzz import fuzz
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, send_file,
    jsonify, current_app, abort
)
from flask_login import current_user
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangDocument
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI

from app.extensions import db
from app.models import Document, UserRole
from .config import (
    PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, NAMESPACE, GEMINI_API_KEY,
    CHAT_PROMPT_TEMPLATE, EMBEDDING_DIMENSION, EMBEDDER, ALLOWED_EXTENSIONS,
    GENERAL_QUERY_PATTERN, FOLLOWUP_POSITIVE, FOLLOWUP_NEGATIVE, FOLLOWUP_NEUTRAL,
    GREETING_KEYWORDS, TIMEZONE, PINECONE_SPEC
)

cardiobot = Blueprint('cardiobot', __name__, url_prefix='/cardiobot')

# Global state
class ChatState:
    def __init__(self):
        self.vector_store: Optional[LangchainPinecone] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.conversation_history: List[Dict] = []
        self.chat_cache: Dict[str, Dict] = {}
        self.last_bot_question: str = ""
        self.conversation_context: Dict[str, Optional[str]] = {
            "last_topic": None,
            "last_question": None
        }

chat_state = ChatState()

def log_response_time(start_time: float, log_prefix: str = "") -> float:
    """Calculate and log response time."""
    response_time = time.time() - start_time
    current_app.logger.info(f"{log_prefix}[RESPONSE TIME] {response_time:.3f} seconds")
    return response_time

def get_pinecone_client() -> Pinecone:
    """Initialize Pinecone client."""
    return Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV, grpc=True)

def initialize_vector_store() -> LangchainPinecone:
    """Initialize or load Pinecone vector store."""
    pc = get_pinecone_client()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(**PINECONE_SPEC)
        )
    return LangchainPinecone(
        index_name=INDEX_NAME,
        embedding=EMBEDDER,
        namespace=NAMESPACE,
        text_key="page_content"
    )

def create_conversational_chain(vector_store: LangchainPinecone) -> ConversationalRetrievalChain:
    """Create conversational chain with Gemini LLM."""
    chat_prompt = PromptTemplate(
        template=CHAT_PROMPT_TEMPLATE,
        input_variables=["question", "context", "chat_history"]
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 5}
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        output_key="answer",
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": chat_prompt,
            "document_variable_name": "context"
        }
    )

def fuzzy_match_keyword(input_text: str, keyword_set: set, threshold: float = 80) -> Optional[str]:
    """Perform fuzzy matching against a set of keywords."""
    input_text = input_text.lower().strip()
    best_match, highest_score = None, threshold
    for keyword in keyword_set:
        score = fuzz.partial_ratio(input_text, keyword)
        if score > highest_score:
            highest_score, best_match = score, keyword
    return best_match if highest_score >= threshold else None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_sentences(answer: str, context: str, max_sent: int = 2) -> List[str]:
    """Extract relevant sentences from context based on answer."""
    sentences = re.split(r'(?<=[\.\!?])\s+', context)
    ans_words = set(re.findall(r'\w+', answer.lower()))
    scored = [(len(ans_words & set(re.findall(r'\w+', s.lower()))), s) for s in sentences if s]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sent]]

def is_based_on_docs(answer: str, doc_contents: List[str], threshold: float = 0.7) -> bool:
    """Check if answer is grounded in documents."""
    try:
        answer_emb = EMBEDDER.embed_query(answer)
        doc_embs = EMBEDDER.embed_documents(doc_contents)
        similarities = cosine_similarity([answer_emb], doc_embs)[0]
        max_score = float(np.max(similarities))
        current_app.logger.info(f"[HALLUCINATION] Max cosine similarity: {max_score:.4f}")
        return max_score >= threshold
    except Exception as e:
        current_app.logger.warning(f"[HALLUCINATION] Embedding failed: {e}")
        return False

def is_doc_relevant_hybrid(
    question: str, doc: str, embedder, min_word_overlap: int = 1,
    min_cosine_similarity: float = 0.7, alpha: float = 0.5
) -> bool:
    """Check document relevance using hybrid approach."""
    q_words = set(re.findall(r'\w+', question.lower()))
    d_words = set(re.findall(r'\w+', doc.lower()))
    word_overlap_score = len(q_words & d_words)
    try:
        question_emb = embedder.embed_query(question)
        doc_emb = embedder.embed_query(doc)
        cosine_sim = cosine_similarity([question_emb], [doc_emb])[0][0]
    except Exception as e:
        current_app.logger.warning(f"[RELEVANCE] Embedding failed: {e}")
        cosine_sim = 0.0
    return word_overlap_score >= min_word_overlap and cosine_sim >= min_cosine_similarity

def classify_followup_response(message: str) -> str:
    """Classify user response as positive, negative, neutral, or unknown."""
    normalized = message.lower().strip()
    tokens = re.findall(r'\b\w+\b', normalized)
    if any(kw in tokens for kw in FOLLOWUP_POSITIVE):
        return "positive"
    if any(kw in tokens for kw in FOLLOWUP_NEGATIVE):
        return "negative"
    if any(kw in tokens for kw in FOLLOWUP_NEUTRAL):
        return "neutral"
    return "unknown"

def estimate_batch_size(batch_docs: List[LangDocument], batch_ids: List[str], embeddings) -> int:
    """Estimate size of batch payload."""
    texts = [chunk.page_content for chunk in batch_docs]
    vectors = embeddings.embed_documents(texts)
    payload = [
        {"id": vid, "values": vec, "metadata": {"source": chunk.metadata["source"]}, "text": text}
        for vid, text, vec, chunk in zip(batch_ids, texts, vectors, batch_docs)
    ]
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

def reload_vector_store() -> bool:
    """Rebuild vector store from MySQL documents."""
    docs = Document.query.all()
    if not docs:
        current_app.logger.info("[RELOAD] MySQL empty, skip rebuild.")
        return False

    pc = get_pinecone_client()
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception as e:
        current_app.logger.warning(f"[RELOAD] Namespace clear failed: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300, length_function=len, separators=["\n\n", "\n", ".", " "]
    )
    docs_for_index = []
    for d in docs:
        if not d.title_file.lower().endswith('.pdf'):
            current_app.logger.info(f"[RELOAD] Skipped non-PDF file: {d.title_file}")
            continue
        try:
            reader = PdfReader(BytesIO(d.file_data))
            text = ''.join(p.extract_text() or '' for p in reader.pages)
        except Exception as e:
            current_app.logger.warning(f"[RELOAD] Failed to extract text from {d.title_file}: {e}")
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs_for_index.append(LangDocument(
                page_content=chunk,
                metadata={"source": d.title_file, "vector_id": f"{d.id}-{i}"}
            ))

    chat_state.vector_store = initialize_vector_store()
    batch_size = 170
    all_ids = [md.metadata["vector_id"] for md in docs_for_index]
    for i in range(0, len(docs_for_index), batch_size):
        batch_docs = docs_for_index[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        size_bytes = estimate_batch_size(batch_docs, batch_ids, EMBEDDER)
        if size_bytes > 4_194_304:
            current_app.logger.warning("⚠️ Batch exceeds 4 MB, reduce BATCH_SIZE!")
        current_app.logger.info(f"[RELOAD] Batch {i//batch_size+1}: {size_bytes/(1024*1024):.2f} MB")
        chat_state.vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    current_app.logger.info(f"[RELOAD] Indexed {len(docs_for_index)} chunks.")
    return True

def initialize(state):
    """Initialize cardiobot components."""
    app = state.app
    chat_state.vector_store = initialize_vector_store()
    chat_state.chain = create_conversational_chain(chat_state.vector_store)
    app.logger.info("✅ Cardiobot ready: vector_store & chain initialized.")

def is_general_document_query(text: str) -> bool:
    """Check if text is a general document query."""
    return bool(GENERAL_QUERY_PATTERN.search(text.lower().strip()))

def reconstruct_with_context(
    user_input: str, history: List[Dict], last_topic: Optional[str],
    last_question: Optional[str], max_short: int = 3
) -> str:
    """Reconstruct user input with context for short queries."""
    ui = user_input.strip().lower()
    current_app.logger.info(f"[RECONSTRUCT] Raw user_input: {ui!r}")
    parts = re.split(r'[.?!]\s*', ui)
    cleaned = [p for p in parts if p]
    core = cleaned[-1] if cleaned else ui
    current_app.logger.info(f"[RECONSTRUCT] Core sentence: {core!r}")
    tokens = core.split()

    if len(tokens) <= max_short:
        if last_question and is_similar_topic(core, last_question, EMBEDDER):
            question_core = re.split(r'[.?!]\s*', last_question.lower())[-1].rstrip('?')
            return f"{core} {question_core}?"
        elif last_topic:
            topic_core = last_topic.lower().rstrip(' ?.!')
            return f"{core} {topic_core}?"
    for entry in reversed(history):
        if not entry.get("followup") and entry.get("user"):
            return f"{entry['user'].strip().lower()} {core}"
    return core

def extract_topic_from_docs(question: str, docs: List, embedder, threshold: float = 0.7) -> Optional[str]:
    """Extract topic from relevant documents."""
    try:
        q_vec = embedder.embed_query(question)
        combined_texts = [
            f"{getattr(d, 'title_file', d.metadata.get('title', ''))}\n{d.page_content}" for d in docs
        ]
        doc_vecs = embedder.embed_documents(combined_texts)
        sims = cosine_similarity([q_vec], doc_vecs)[0]
        if float(np.max(sims)) < threshold:
            return None
        best_idx = int(np.argmax(sims))
        best_doc = docs[best_idx]
        raw_title = getattr(best_doc, 'title_file', best_doc.metadata.get("title", ""))
        topic = re.sub(r"^(pedoman|panduan)\s*", "", raw_title.strip(), flags=re.IGNORECASE)
        return topic.strip()
    except Exception as e:
        current_app.logger.warning(f"[TOPIC EXTRACTION] Failed: {e}")
        return None

def is_similar_topic(new_text: str, last_text: str, embedder, threshold: float = 0.6) -> bool:
    """Check if two texts are on similar topics."""
    try:
        vec_new = embedder.embed_query(new_text)
        vec_last = embedder.embed_query(last_text)
        sim = cosine_similarity([vec_new], [vec_last])[0, 0]
        current_app.logger.info(f"[SIMILARITY] Cosine similarity: {sim:.4f}")
        return sim >= threshold
    except Exception as e:
        current_app.logger.warning(f"[SIMILARITY] Failed: {e}")
        return False

def clean_answer(text: str) -> str:
    """Clean and normalize answer text."""
    text = re.sub(r'\n{2,}', '\n\n', text).replace("**", "").replace("*", "")
    text = re.sub(r'\.{3,}\s*\d+', '', text)
    text = re.sub(r'^(BAB\s+[IVXLC]+|[0-9]+\.[0-9]*)\.\s*', '', text, flags=re.MULTILINE)
    return re.sub(r'\n{2,}', '\n\n', text).strip()

def clear_chat_cache():
    """Clear chat cache and reset context."""
    chat_state.chat_cache.clear()
    chat_state.conversation_context.update({"last_topic": None, "last_question": None})
    current_app.logger.info("✅ [DEBUG] Chat cache cleared and context reset")

@cardiobot.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear chat cache endpoint."""
    start_time = time.time()
    clear_chat_cache()
    response_time = log_response_time(start_time, "[CLEAR_CACHE] ")
    return jsonify({"status": "ok", "response_time": response_time}), 200

@cardiobot.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    raw = data.get("message", "").strip()
    if not raw:
        response_time = log_response_time(start_time, "[CHAT] ")
        return jsonify({"error": "Empty input.", "response_time": response_time}), 400

    user_input = re.sub(r"\s+", " ", raw).lower()
    core = re.split(r'[.?!]\s*', user_input)[-1]
    current_app.logger.info(f"[CHAT] Received message: {user_input!r}")
    
    if fuzzy_match_keyword(user_input, GREETING_KEYWORDS):
        response = {
            "answer": "Halo! Saya adalah Cardiobot, siap membantu dengan pertanyaan seputar kardiovaskular.",
            "sources": [],
            "response_time": log_response_time(start_time, "[CHAT] ")
        }
        chat_state.chat_cache[user_input] = response
        chat_state.conversation_history.append({"user": raw, "bot": response["answer"], "followup": False, "category": "greeting"})
        current_app.logger.info(f"[CHAT] Greeting detected: {user_input!r}")
        return jsonify(response)

    docs = Document.query.all()
    if not docs:
        response = {
            "answer": "Maaf, saat ini tidak ada dokumen yang tersedia.",
            "sources": [],
            "based_on_docs": True,
            "response_time": log_response_time(start_time, "[CHAT] ")
        }
        chat_state.chat_cache[user_input] = response
        chat_state.conversation_history.append({"user": raw, "bot": response["answer"], "followup": False, "category": "no_docs"})
        current_app.logger.info(f"[CHAT] No documents available in database.")
        return jsonify(response)

    if not chat_state.vector_store or not chat_state.chain:
        response_time = log_response_time(start_time, "[CHAT] ")
        return jsonify({"error": "Bot not ready.", "response_time": response_time}), 400

    if user_input in chat_state.chat_cache:
        response = chat_state.chat_cache[user_input].copy()
        response["response_time"] = log_response_time(start_time, "[CHAT] ")
        current_app.logger.info(f"[CHAT] Cache HIT for key {user_input!r}")
        return jsonify(response)

    if is_general_document_query(user_input):
        current_app.logger.info(f"[CHAT] Detected general document query: {user_input!r}")
        topics = [doc.title_file.split('.')[0].replace('_', ' ').title() for doc in docs]
        max_initial_topics = 5
        summary = "Dokumen yang tersedia berisi informasi tentang:\n" + "\n".join(f"- {t}" for t in topics[:max_initial_topics])
        if len(topics) > max_initial_topics:
            summary += f"\n- Dan lainnya (+{len(topics[max_initial_topics:])})"
        response = {
            "answer": summary,
            "sources": [],
            "all_topics": topics,
            "response_time": log_response_time(start_time, "[CHAT] ")
        }
        chat_state.chat_cache[user_input] = response
        chat_state.conversation_history.append({"user": raw, "bot": summary, "followup": False, "category": "general"})
        return jsonify(response)

    try:
        is_followup = bool(chat_state.last_bot_question) or len(core.split()) <= 3
        category = classify_followup_response(user_input)
        last_topic = chat_state.conversation_context.get("last_topic")
        last_question = chat_state.conversation_context.get("last_question")
        current_app.logger.info(f"[CHAT] Followup: {is_followup}, Category: {category}, Last Topic: {last_topic}, Last Question: {last_question}")

        if is_followup and category == "negative":
            response = {
                "answer": "Baik, terima kasih. Silakan ajukan pertanyaan lain! ☺️.",
                "sources": [],
                "response_time": log_response_time(start_time, "[CHAT] ")
            }
            chat_state.chat_cache[user_input] = response
            chat_state.conversation_history.append({"user": raw, "bot": response["answer"], "followup": True, "category": category})
            chat_state.last_bot_question = ""
            current_app.logger.info(f"[CHAT] Negative followup response: {user_input!r}")
            return jsonify(response)

        if is_followup and category == "neutral":
            response = {
                "answer": "Baik, saya akan coba jelaskan lagi. Apa yang masih belum jelas? 🤔",
                "sources": [],
                "response_time": log_response_time(start_time, "[CHAT] ")
            }
            chat_state.chat_cache[user_input] = response
            chat_state.conversation_history.append({"user": raw, "bot": response["answer"], "followup": True, "category": category})
            chat_state.last_bot_question = ""
            current_app.logger.info(f"[CHAT] Neutral followup response: {user_input!r}")
            return jsonify(response)

        if is_followup and category == "positive":
            follow_q = re.sub(r"(?i)^apakah anda ingin tahu\s*lebih\s*lanjut\s*tentang\s*", "", chat_state.last_bot_question).strip().rstrip('?').lower()
            user_input = follow_q
            current_app.logger.info(f"[FOLLOWUP POSITIVE] Query reconstructed: {user_input!r}")

        elif is_followup and category == "unknown" and (last_topic or last_question):
            if last_question and is_similar_topic(core, last_question, EMBEDDER):
                user_input = reconstruct_with_context(user_input, chat_state.conversation_history, last_topic, last_question)
                current_app.logger.info(f"[FOLLOWUP UNKNOWN] Reconstructed with last question: {user_input!r}")
            elif last_topic and is_similar_topic(core, last_topic, EMBEDDER):
                user_input = reconstruct_with_context(user_input, chat_state.conversation_history, last_topic, last_question)
                current_app.logger.info(f"[FOLLOWUP UNKNOWN] Reconstructed with last topic: {user_input!r}")
            else:
                is_followup = False
                current_app.logger.info(f"[FOLLOWUP UNKNOWN] No reconstruction, treated as new query")

        res = chat_state.chain.invoke({"question": user_input})
        docs = res.get("source_documents", [])
        answer = clean_answer(res["answer"])
        current_app.logger.info(f"[CHAT] Raw LLM answer: {answer!r}")

        topic = extract_topic_from_docs(user_input, docs, EMBEDDER)
        if topic:
            chat_state.conversation_context["last_topic"] = topic.lower()
            current_app.logger.info(f"[CHAT] Extracted topic: {topic}")

        chat_state.conversation_context["last_question"] = user_input

        doc_contents = [d.page_content for d in docs]
        hallucinated = not is_based_on_docs(answer, doc_contents)
        no_info_response = "maaf" in answer.lower() and "tidak ditemukan" in answer.lower()
        current_app.logger.info(f"[CHAT] Hallucinated: {hallucinated}, No info response: {no_info_response}")

        if hallucinated or no_info_response or not any(is_doc_relevant_hybrid(user_input, doc, EMBEDDER) for doc in doc_contents):
            answer = "Maaf, saya tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan lebih spesifik! ✍️😊"
            snippets = []
            chat_state.last_bot_question = ""
            current_app.logger.info(f"[CHAT] Answer rejected: {'hallucinated' if hallucinated else 'no info or irrelevant'}")
        else:
            parts = answer.rsplit("\n\n", 1)
            chat_state.last_bot_question = parts[1].strip() if len(parts) == 2 and "?" in parts[1] else ""
            valid_docs = [d for d in docs if is_doc_relevant_hybrid(user_input, d.page_content, EMBEDDER)]
            snippets = [
                {"file": d.metadata.get("source", "Unknown"), "text": s}
                for d in valid_docs
                for s in (extract_sentences(answer, d.page_content) or [d.page_content[:200]])
            ]
            current_app.logger.info(f"[CHAT] Valid snippets generated: {len(snippets)}")

        response = {
            "answer": answer,
            "sources": snippets,
            "based_on_docs": not hallucinated,
            "response_time": log_response_time(start_time, "[CHAT] ")
        }
        chat_state.chat_cache[user_input] = response
        chat_state.conversation_history.append({
            "user": raw,
            "bot": answer,
            "followup": is_followup,
            "category": category,
            "based_on_docs": not hallucinated
        })
        current_app.logger.info(f"[CHAT] Generated answer: {answer!r}")
        current_app.logger.info(f"[CHAT] Source snippets ({len(snippets)}): {json.dumps(snippets, ensure_ascii=False)}")
        current_app.logger.info(f"[CHAT] Cache MISS: stored response under key {user_input!r}")
        current_app.logger.info(f"[CHAT] Current cache keys: {list(chat_state.chat_cache.keys())}")
        current_app.logger.info(f"[CHAT] Based on docs: {not hallucinated}")
        current_app.logger.info(f"[CHAT] Response time: {response['response_time']:.3f} seconds")
        return jsonify(response)

    except Exception as e:
        current_app.logger.error(f"[ERROR] Chat error: {e}", exc_info=True)
        response_time = log_response_time(start_time, "[CHAT] ")
        return jsonify({"error": "Server error.", "response_time": response_time}), 500

@cardiobot.route('/live-chat', defaults={'file_id': None})
@cardiobot.route('/live-chat/<int:file_id>')
def live_chat(file_id):
    """Render live chat interface."""
    documents = Document.query.all()
    current = Document.query.get(file_id) if file_id and documents else documents[0] if documents else None
    return render_template(
        'main/live_chat.html',
        documents=documents,
        document=current,
        navbar_title='Cardiobot'
    )

@cardiobot.route('/documents/preview/<int:file_id>')
def preview_file(file_id):
    """Serve file preview."""
    start_time = time.time()
    doc = Document.query.get_or_404(file_id)
    response = send_file(
        BytesIO(doc.file_data),
        download_name=doc.title_file,
        as_attachment=False,
        mimetype='application/pdf'
    )
    log_response_time(start_time, "[PREVIEW] ")
    return response

@cardiobot.route('/documents/view/<int:file_id>')
def view_document(file_id):
    """Render document view."""
    start_time = time.time()
    doc = Document.query.get_or_404(file_id)
    mime = doc.title_file.rsplit('.', 1)[1].lower()
    preview_url = url_for('.preview_file', file_id=file_id)
    response = render_template(
        'admin/view_document.html',
        navbar_title='Documents View',
        document=doc,
        mime_type=f'application/{mime}',
        preview_url=preview_url
    )
    log_response_time(start_time, "[VIEW_DOCUMENT] ")
    return response

@cardiobot.route('/documents/upload', methods=['POST'])
def upload_documents():
    """Upload and index documents."""
    start_time = time.time()
    if current_user.role != UserRole == 'admin':
        abort(403)
    
    files = request.files.getlist('files[]')
    if not files or not all(allowed_file(f.filename) for f in files):
        flash("Invalid or no files provided.", "error")
        log_response_time(start_time, "[UPLOAD] ")
        return redirect(url_for('admin.settings', tab='docs'))

    for file in files:
        name = secure_filename(file.filename).lower()
        db.session.add(Document(
            title_file=name,
            file_data=file.read(),
            user_id=current_user.id,
            created_at=datetime.now(TIMEZONE),
            updated_at=datetime.now(TIMEZONE)
        ))
    db.session.commit()

    success = reload_vector_store()
    if success:
        chat_state.chain = create_conversational_chain(chat_state.vector_store)

    clear_chat_cache()
    
    pc = get_pinecone_client()
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    total_docs = Document.query.count()
    remaining_vectors = stats.get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)
    
    flash(f"✅ Indexed {total_docs} documents, {remaining_vectors} vectors.", "success")
    log_response_time(start_time, "[UPLOAD] ")
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/documents/delete/<int:file_id>', methods=['POST'])
def delete_document(file_id):
    """Delete a document and update index."""
    start_time = time.time()
    if current_user.role != UserRole == 'admin':
        abort(403)

    pc = get_pinecone_client()
    index = pc.Index(INDEX_NAME)
    
    doc = Document.query.get_or_404(file_id)
    filename = doc.title_file.lower()
    total_docs_before = Document.query.count()
    stats_before = index.describe_index_stats().get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)

    if filename.endswith('.pdf'):
        try:
            reader = PdfReader(BytesIO(doc.file_data))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300, separators=["\n\n", "\n", ".", " "]
            )
            vector_ids = [f"{file_id}-{i}" for i in range(len(splitter.split_text(text)))]
            index.delete(ids=vector_ids, namespace=NAMESPACE)
        except Exception as e:
            current_app.logger.warning(f"[DELETE] Failed to process vectors for {file_id}: {e}")

    db.session.delete(doc)
    db.session.commit()

    reload_vector_store()
    chat_state.chain = create_conversational_chain(chat_state.vector_store)
    clear_chat_cache()

    total_docs_after = Document.query.count()
    stats_after = index.describe_index_stats().get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)
    flash(f"✅ Deleted: {total_docs_before} -> {total_docs_after} docs, {stats_before} -> {stats_after} vectors.", "success")
    log_response_time(start_time, "[DELETE] ")
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/delete_all_documents', methods=['POST'])
def delete_all_documents():
    """Delete all documents and reset index."""
    start_time = time.time()
    if current_user.role != UserRole == 'admin':
        abort(403)

    Document.query.delete()
    db.session.commit()

    pc = get_pinecone_client()
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception as e:
        current_app.logger.warning(f"[DELETE_ALL] Namespace cleanup failed: {e}")

    chat_state.vector_store = initialize_vector_store()
    chat_state.chain = create_conversational_chain(chat_state.vector_store)
    clear_chat_cache()

    total_docs = Document.query.count()
    stats = index.describe_index_stats().get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)
    flash(f"✅ Deleted all: {total_docs} documents, {stats} vectors remain.", "success")
    log_response_time(start_time, "[DELETE_ALL] ")
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Retrieve conversation history."""
    start_time = time.time()
    response = jsonify({"history": chat_state.conversation_history})
    response_time = log_response_time(start_time, "[CHAT_HISTORY] ")
    return response