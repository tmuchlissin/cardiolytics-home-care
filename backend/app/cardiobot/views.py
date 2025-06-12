import os
import json
import numpy as np
import time
import docx2txt
import re
from rapidfuzz import fuzz
from io import BytesIO
from datetime import datetime
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, send_file, jsonify, current_app,
    abort
)
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangDocument
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI

from app.extensions import db
from app.models import Document

from .config import (
    PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, NAMESPACE, GEMINI_API_KEY, CHAT_PROMPT_TEMPLATE,
    EMBEDDING_DIMENSION, EMBEDDER, ALLOWED_EXTENSIONS, GENERAL_QUERY_PATTERN,
    FOLLOWUP_POSITIVE, FOLLOWUP_NEGATIVE, FOLLOWUP_NEUTRAL, GREETING_KEYWORDS, TIMEZONE
)

load_dotenv()

cardiobot = Blueprint('cardiobot', __name__, url_prefix='/cardiobot')

# In-memory state
conversation_history: list[dict] = []
chat_cache: dict[str, dict] = {}
vector_store = None
chain = None
last_bot_question = ""
conversation_context = {
    "last_topic": None,
    "last_question": None
}

def init_pinecone():
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        grpc=True
    )
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=spec
        )
    return pc

def fuzzy_match_keyword(input_text: str, keyword_set: set, threshold: float = 80) -> str:
    input_text = input_text.lower().strip()
    best_match = None
    highest_score = threshold
    for keyword in keyword_set:
        score = fuzz.partial_ratio(input_text, keyword)
        if score > highest_score:
            highest_score = score
            best_match = keyword
    return best_match if highest_score >= threshold else input_text

def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def extract_sentences(answer: str, context: str, max_sent=2) -> list[str]:
    sentences = re.split(r'(?<=[\.\!?])\s+', context)
    ans_words = set(re.findall(r'\w+', answer.lower()))
    scored = []
    for sent in sentences:
        words = set(re.findall(r'\w+', sent.lower()))
        score = len(ans_words & words)
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sent]]

def is_based_on_docs(answer: str, doc_contents: list[str], threshold: float = 0.7) -> bool:
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
        question: str, doc: str, EMBEDDER,
        min_word_overlap: int = 1, min_cosine_similarity: float = 0.7, alpha: float = 0.5
) -> bool:
    q_words = set(re.findall(r'\w+', question.lower()))
    d_words = set(re.findall(r'\w+', doc.lower()))
    word_overlap_score = len(q_words & d_words)
    try:
        question_emb = EMBEDDER.embed_query(question)
        doc_emb = EMBEDDER.embed_query(doc)
        cosine_sim = cosine_similarity([question_emb], [doc_emb])[0][0]
    except Exception as e:
        current_app.logger.warning(f"[RELEVANCE] Embedding failed: {e}")
        cosine_sim = 0.0
    overlap_pass = word_overlap_score >= min_word_overlap
    similarity_pass = cosine_sim >= min_cosine_similarity
    return overlap_pass and similarity_pass

def classify_followup_response(message: str) -> str:
    normalized = message.lower().strip()
    tokens = re.findall(r"\b\w+\b", normalized)
    for kw in FOLLOWUP_POSITIVE:
        if kw in tokens:
            return "positive"
    for kw in FOLLOWUP_NEGATIVE:
        if kw in tokens:
            return "negative"
    for kw in FOLLOWUP_NEUTRAL:
        if kw in tokens:
            return "neutral"
    return "unknown"

def estimate_batch_size(batch_docs, batch_ids, embeddings):
    texts = [chunk.page_content for chunk in batch_docs]
    vectors = embeddings.embed_documents(texts)
    payload = []
    for vid, text, vec, chunk in zip(batch_ids, texts, vectors, batch_docs):
        payload.append({
            "id": vid,
            "values": vec,
            "metadata": {"source": chunk.metadata["source"]},
            "text": text
        })
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return len(raw)

def load_or_initialize_vector_store(app=None) -> LangchainPinecone | None:
    global vector_store
    
    if vector_store:
        return vector_store
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        grpc=True
    )
    if INDEX_NAME not in pc.list_indexes().names():
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=spec
        )
    vector_store = LangchainPinecone(
        index_name=INDEX_NAME,
        embedding=EMBEDDER,
        namespace=NAMESPACE,
        text_key="page_content",
    )
    return vector_store

def reload_vector_store():
    global vector_store
    
    docs = Document.query.all()
    if not docs:
        current_app.logger.info("[reload] MySQL empty, skip rebuild.")
        return False
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        grpc=True
    )
    try:
        index_list = pc.list_indexes().names()
        if INDEX_NAME not in index_list:
            raise RuntimeError(f"Index '{INDEX_NAME}' not found in Pinecone. Available: {index_list}")
    except Exception as e:
        current_app.logger.error(f"[Pinecone] Could not verify index: {e}")
        return False
    idx = pc.Index(INDEX_NAME)
    try:
        idx.delete(delete_all=True, namespace=NAMESPACE)
    except Exception:
        current_app.logger.warning("[reload] Namespace clear failed, continuing.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    docs_for_index = []
    
    for d in docs:
        data = d.file_data
        text = ''
        if d.title_file.lower().endswith('.pdf'):
            reader = PdfReader(BytesIO(data))
            text = ''.join(p.extract_text() or '' for p in reader.pages)
        elif d.title_file.lower().endswith('.docx'):
            text = docx2txt.process(BytesIO(data))
        else:
            text = data.decode('utf-8', errors='ignore')
        for i, chunk in enumerate(splitter.split_text(text)):
            docs_for_index.append(LangDocument(
                page_content=chunk,
                metadata={"source": d.title_file, "vector_id": f"{d.id}-{i}"}
            ))
            
    vector_store = LangchainPinecone(
        index_name=INDEX_NAME,
        embedding=EMBEDDER,
        namespace=NAMESPACE,
        text_key="page_content",
    )
    
    BATCH_SIZE = 170
    all_ids = [md.metadata["vector_id"] for md in docs_for_index]
    for i in range(0, len(docs_for_index), BATCH_SIZE):
        batch_docs = docs_for_index[i: i + BATCH_SIZE]
        batch_ids = all_ids[i: i + BATCH_SIZE]
        size_bytes = estimate_batch_size(batch_docs, batch_ids, EMBEDDER)
        mb = size_bytes / (1024 * 1024)
        if size_bytes > 4_194_304:
            print("⚠️ Exceeds 4 MB, reduce BATCH_SIZE!")
        current_app.logger.info(f"[reload] Batch {i//BATCH_SIZE+1}: {mb:.2f} MB")
        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )
    current_app.logger.info(f"[reload] Indexed {len(docs_for_index)} chunks.")
    return True

def create_conversational_chain(vs: LangchainPinecone) -> ConversationalRetrievalChain:
    chat_prompt = PromptTemplate(
        template=CHAT_PROMPT_TEMPLATE,
        input_variables=["question", "context", "chat_history"]
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    # Use ConversationBufferMemory by default, with option for ConversationSummaryMemory
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     output_key="answer"
    # )
    # Optional: Switch to ConversationSummaryMemory for long conversations
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    retriever = vs.as_retriever(
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

def initialize(state):
    global vector_store, chain
    app = state.app
    vs = load_or_initialize_vector_store(app)
    if not vs:
        app.logger.warning("⚠️ Pinecone index does not exist yet.")
        return
    chain = create_conversational_chain(vs)
    app.logger.info("✅ Cardiobot ready: vector_store & chain initialized.")

@cardiobot.route('/clear_cache', methods=['POST'])
def clear_cache():
    global chat_cache, conversation_context
    chat_cache.clear()
    conversation_context["last_topic"] = None
    conversation_context["last_question"] = None
    current_app.logger.info("✅ [DEBUG] chat_cache cleared and context reset")
    return jsonify({"status": "ok"}), 200

def is_general_document_query(text: str) -> bool:
    return bool(GENERAL_QUERY_PATTERN.search(text.lower().strip()))

def reconstruct_with_context(user_input: str, history: list, last_topic: str = None, last_question: str = None, max_short: int = 3) -> str:
    ui = user_input.strip().lower()
    current_app.logger.info(f"[RECONSTRUCT] Raw user_input: {ui!r}")
    parts = re.split(r'[.?!]\s*', ui)
    cleaned = [p for p in parts if p]
    core = cleaned[-1] if cleaned else ui
    current_app.logger.info(f"[RECONSTRUCT] Core sentence: {core!r}")
    tokens = core.split()
    current_app.logger.info(f"[RECONSTRUCT] Token count: {len(tokens)} (max_short={max_short})")

    if len(tokens) <= max_short:
        if last_question and is_similar_topic(core, last_question, EMBEDDER, threshold=0.6):
            question_core = re.split(r'[.?!]\s*', last_question.lower())[-1].rstrip('?')
            merged = f"{core} {question_core}?"
            current_app.logger.info(f"[RECONSTRUCT] Merged with last_question: {merged!r}")
            return merged
        
        elif last_topic:
            topic_core = last_topic.lower().rstrip(' ?.!')
            merged = f"{core} {topic_core}?"
            current_app.logger.info(f"[RECONSTRUCT] Merged with last_topic: {merged!r}")
            return merged
    
    for entry in reversed(history):
        if not entry.get("followup") and entry.get("user"):
            merged = f"{entry['user'].strip().lower()} {core}"
            current_app.logger.info(f"[RECONSTRUCT] Merged with history: {merged!r}")
            return merged
    current_app.logger.info(f"[RECONSTRUCT] Returning core only")
    return core

def extract_topic_from_docs(question: str, docs: list, EMBEDDER, threshold: float = 0.7):
    try:
        q_vec = EMBEDDER.embed_query(question)
        combined_texts = [
            f"{getattr(d, 'title_file', d.metadata.get('title', ''))}\n{d.page_content}" for d in docs
        ]
        doc_vecs = EMBEDDER.embed_documents(combined_texts)
        sims = cosine_similarity([q_vec], doc_vecs)[0]
        max_score = float(np.max(sims))
        if max_score < threshold:
            return None
        best_idx = int(np.argmax(sims))
        best_doc = docs[best_idx]
        raw_title = getattr(best_doc, 'title_file', best_doc.metadata.get("title", ""))
        topic = re.sub(r"^(pedoman|panduan)\s*", "", raw_title.strip(), flags=re.IGNORECASE)
        return topic.strip()
    except Exception as e:
        current_app.logger.warning(f"[TOPIC EXTRACTION] Failed: {e}")
        return None

def is_similar_topic(new_text: str, last_text: str, EMBEDDER, threshold: float = 0.6) -> bool:
    try:
        vec_new = EMBEDDER.embed_query(new_text)
        vec_last = EMBEDDER.embed_query(last_text)
        sim = cosine_similarity([vec_new], [vec_last])[0, 0]
        current_app.logger.info(f"[SIMILARITY] Cosine similarity between '{new_text}' and '{last_text}': {sim:.4f}")
        return sim >= threshold
    except Exception as e:
        current_app.logger.warning(f"[SIMILARITY] Failed: {e}")
        return False

def clean_answer(text):
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'\.{3,}\s*\d+', '', text)
    text = re.sub(r'^(BAB\s+[IVXLC]+|[0-9]+\.[0-9]*)\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    return text

@cardiobot.route('/chat', methods=['POST'])
def chat():
    global chat_cache, chain, vector_store, conversation_history, last_bot_question, conversation_context
    
    start_time = time.time()
    
    data = request.get_json(silent=True) or {}
    raw = data.get("message", "") or ""
    user_input = re.sub(r"\s+", " ", raw).strip().lower()
    core = re.split(r'[.?!]\s*', user_input)[-1]
    current_app.logger.info(f"[CHAT] Received message: {user_input!r}")

    if not vector_store or not chain or not user_input:
        current_app.logger.warning("[CHAT] Bot not ready or input is empty")
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify({"error": "Bot is not ready or input is empty."}), 400

    if user_input in chat_cache:
        current_app.logger.info(f"[CHAT] Cache HIT for key {user_input!r}")
        response = chat_cache[user_input]
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify(response)

    matched_greeting = fuzzy_match_keyword(user_input, GREETING_KEYWORDS)
    if matched_greeting:
        resp = {
            "answer": "Halo! Saya adalah Cardiobot. Saya hanya akan menjawab pertanyaan seputar kardiovaskular.",
            "sources": []
        }
        chat_cache[user_input] = resp
        current_app.logger.info(f"[CHAT] Greeting stored to cache under key {user_input!r}")
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify(resp)

    if is_general_document_query(user_input):
        current_app.logger.info(f"[CHAT] General question detected: {user_input}")
        docs = Document.query.all()
        if not docs:
            response = {
                "answer": "Maaf, saat ini tidak ada dokumen yang tersedia.",
                "sources": [],
                "based_on_docs": True
            }
            
            chat_cache[user_input] = response
            conversation_history.append({
                "user": raw.strip(),
                "bot": response["answer"],
                "followup": False,
                "category": "general",
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
            return jsonify(response)
        
        topics = [doc.title_file.split('.')[0].replace('_', ' ').title() for doc in docs]
        max_initial_topics = 5
        initial_topics = topics[:max_initial_topics]
        remaining_topics = topics[max_initial_topics:]
        remaining_count = len(remaining_topics)
        
        summary = "Dokumen yang tersedia berisi informasi tentang:\n"
        summary += "\n".join(f"- {topic}" for topic in initial_topics)
        
        if remaining_count > 0:
            summary += f"\n- Dan lainnya (+{remaining_count})"
        response = {
            "answer": summary,
            "sources": [],
            "all_topics": topics
        }
        
        chat_cache[user_input] = response
        conversation_history.append({
            "user": raw.strip(),
            "bot": summary,
            "followup": False,
            "category": "general",
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify(response)

    try:
        is_followup = bool(last_bot_question) or len(core.split()) <= 3
        category = classify_followup_response(user_input)
        last_t = conversation_context.get("last_topic")
        last_q = conversation_context.get("last_question")

        if is_followup and category == "negative":
            response = {
                "answer": "Baik, terima kasih. Jika ada pertanyaan lain, silakan ajukan! ☺️.",
                "sources": []
            }
            chat_cache[user_input] = response
            conversation_history.append({
                "user": raw.strip(),
                "bot": response["answer"],
                "followup": True,
                "category": category
            })
            last_bot_question = ""
            end_time = time.time()
            response_time = end_time - start_time
            current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
            return jsonify(response)

        elif is_followup and category == "neutral":
            response = {
                "answer": "Baik, saya akan coba menjelaskan dengan lebih sederhana.\n\n🤔 Apa yang masih belum jelas bagi Anda?",
                "sources": []
            }
            chat_cache[user_input] = response
            conversation_history.append({
                "user": raw.strip(),
                "bot": response["answer"],
                "followup": True,
                "category": category
            })
            last_bot_question = ""
            end_time = time.time()
            response_time = end_time - start_time
            current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
            return jsonify(response)

        elif is_followup and category == "positive":
            follow_q = last_bot_question or ""
            follow_q = re.sub(
                r"(?i)^apakah anda ingin tahu lebih lanjut tentang\s*",
                "",
                follow_q
            ).strip().rstrip('?').lower()
            user_input = follow_q
            current_app.logger.info(f"[FOLLOWUP POSITIVE] Query for chain: {user_input}")

        elif is_followup and category == "unknown" and (last_t or last_q):
            if last_q and is_similar_topic(core, last_q, EMBEDDER, threshold=0.6):
                user_input = reconstruct_with_context(user_input, conversation_history, last_t, last_q)
                current_app.logger.info(f"[RECONSTRUCT] Reconstructed query: {user_input!r}")
            elif last_t and is_similar_topic(core, last_t, EMBEDDER, threshold=0.6):
                user_input = reconstruct_with_context(user_input, conversation_history, last_t, last_q)
                current_app.logger.info(f"[RECONSTRUCT] Reconstructed query: {user_input!r}")
            else:
                is_followup = False
                current_app.logger.info(f"[CHAT] Low similarity to last_topic/last_question → new topic")

        res = chain.invoke({"question": user_input})
        docs = res.get("source_documents", [])
        answer = clean_answer(res["answer"])

        topic = extract_topic_from_docs(user_input, docs, EMBEDDER, threshold=0.7)
        
        if topic:
            conversation_context["last_topic"] = topic.lower()
            current_app.logger.info(f"[CONTEXT] last_topic set: {conversation_context['last_topic']}")
        
        conversation_context["last_question"] = user_input
        current_app.logger.info(f"[CONTEXT] last_question set: {user_input}")

        doc_contents = [d.page_content for d in docs]
        hallucinated = not is_based_on_docs(answer, doc_contents, threshold=0.7)
        no_info_response = "maaf, informasi" in answer.lower() and "tidak ditemukan" in answer.lower()
        relevant_docs = any(is_doc_relevant_hybrid(user_input, doc, EMBEDDER) for doc in doc_contents)

        if hallucinated or no_info_response or not relevant_docs:
            answer = (
                "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen yang tersedia. 😔\n\n"
                "Silakan ajukan pertanyaan yang lebih spesifik atau tambahkan detail lainnya, ya! ✍️😊"
            )
            snippets = []
            last_bot_question = ""
            if hallucinated:
                current_app.logger.info("[CHAT] Answer rejected due to low similarity score.")
            elif no_info_response:
                current_app.logger.info("[CHAT] Answer rejected due to no-info response.")
            else:
                current_app.logger.info("[CHAT] Answer rejected due to irrelevant documents.")
        else:
            parts = answer.rsplit("\n\n", 1)
            last_bot_question = parts[1].strip() if len(parts) == 2 and "?" in parts[1] else ""
            valid_docs = [d for d in docs if is_doc_relevant_hybrid(user_input, d.page_content, EMBEDDER)]
            snippets = [
                {"file": d.metadata.get("source", "Unknown"), "text": s}
                for d in valid_docs
                for s in (extract_sentences(answer, d.page_content) or [d.page_content[:200]])
            ]

        out = {
            "answer": answer,
            "sources": snippets,
            "based_on_docs": not hallucinated
        }
        chat_cache[user_input] = out
        conversation_history.append({
            "user": raw.strip(),
            "bot": answer,
            "followup": is_followup,
            "category": category,
            "based_on_docs": not hallucinated
        })
        current_app.logger.info(f"[CHAT] Generated answer: {answer!r}")
        current_app.logger.info(f"[CHAT] Source snippets ({len(snippets)}): {json.dumps(snippets)}")
        current_app.logger.info(f"[CHAT] Cache MISS: stored response under key {user_input!r}")
        current_app.logger.info(f"[CHAT] Current cache keys: {list(chat_cache.keys())}")
        current_app.logger.info(f"[CHAT] Based on docs: {not hallucinated}")
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify(out)

    except Exception as e:
        current_app.logger.error("Error in /chat: %s", e, exc_info=True)
        end_time = time.time()
        response_time = end_time - start_time
        current_app.logger.info(f"[CHAT] Response time: {response_time:.3f} seconds")
        return jsonify({"error": "An error occurred on the server."}), 500

@cardiobot.route('/live-chat', defaults={'file_id': None})
@cardiobot.route('/live-chat/<int:file_id>')
def live_chat(file_id):
    documents = Document.query.all()
    if not documents:
        return render_template(
            'main/live_chat.html',
            documents=[],
            document=None,
            navbar_title='Cardiobot'
        )
    current = Document.query.get(file_id) if file_id else documents[0]
    return render_template(
        'main/live_chat.html',
        documents=documents,
        document=current,
        navbar_title='Cardiobot'
    )

@cardiobot.route('/documents/preview/<int:file_id>')
def preview_file(file_id):
    doc = Document.query.get_or_404(file_id)
    return send_file(
        BytesIO(doc.file_data),
        download_name=doc.title_file,
        as_attachment=False,
        mimetype='application/pdf'   
    )

@cardiobot.route('/documents/view/<int:file_id>')
def view_document(file_id):
    doc = Document.query.get_or_404(file_id)
    mime = doc.title_file.rsplit('.',1)[1].lower()
    preview_url = url_for('cardiobot.preview_file', file_id=file_id)
    return render_template(
        'admin/view_document.html',
        navbar_title='Documents View',
        document=doc,
        mime_type=f"application/{mime}",
        preview_url=preview_url,    
    )

@cardiobot.route('/documents', methods=['POST'])
def upload_documents():
    global vector_store, chain      
    
    files = request.files.getlist('files[]')
    
    if not files or not all(allowed_file(f.filename) for f in files):
        flash("❌ Invalid or no files provided.", "error")
        return redirect(url_for('admin.settings', tab='docs'))

    for f in files:
        name = secure_filename(os.path.basename(f.filename)).lower()
        data = f.read()
        db.session.add(Document(
            title_file=name,
            file_data=data,
            created_at=datetime.now(TIMEZONE),
            updated_at=datetime.now(TIMEZONE)
        ))
    db.session.commit()
    success = reload_vector_store()
    if success:
        global chain
        chain = create_conversational_chain(vector_store)
    clear_cache()
    
    total_docs = Document.query.count()
    pc = Pinecone(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENV,
        grpc=True)
    
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    namespace_stats = stats.get('namespaces', {}).get(NAMESPACE, {})
    remaining_vectors = namespace_stats.get('vector_count', 0)  
    
    current_app.logger.info(f"[DEBUG] MySQL docs={total_docs}, Pinecone vectors={remaining_vectors}")
    
    flash(
        f"✅ Indexing complete! MySQL documents: {total_docs} documents, Pinecone vectors: {remaining_vectors} vectors.",
        "success"
    )

    
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/documents/delete/<int:file_id>', methods=['POST'])
def delete_document(file_id):
    pc = Pinecone(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENV,
        grpc=True)
    
    index = pc.Index(INDEX_NAME)
    total_docs_before = Document.query.count()
    stats_before = index.describe_index_stats()
    namespace_stats_before = stats_before.get('namespaces', {}).get(NAMESPACE, {})
    total_vec_before = namespace_stats_before.get('vector_count', 0)
    
    current_app.logger.info(
        f"[DEBUG][BEFORE DELETE] File {file_id}: "
        f"MySQL docs={total_docs_before}, Pinecone vec total={total_vec_before}"
    )
    
    doc = Document.query.get_or_404(file_id)
    raw = doc.file_data
    db.session.delete(doc)
    db.session.commit()
    
    if doc.title_file.lower().endswith('.pdf'):
        reader = PdfReader(BytesIO(raw))
        text = "".join(page.extract_text() or "" for page in reader.pages)
    elif doc.title_file.lower().endswith('.docx'):
        text = docx2txt.process(BytesIO(raw))
    else:
        text = raw.decode('utf-8', errors='ignore')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    
    n_chunks = len(splitter.split_text(text))
    vector_ids = [f"{file_id}-{i}" for i in range(n_chunks)]
    
    try:
        index.delete(ids=vector_ids, namespace=NAMESPACE)
    except Exception as e:
        current_app.logger.warning(f"[DEBUG] Failed to delete vectors for doc {file_id}: {e}")
    
    reload_vector_store()
    
    global chain
    
    chain = create_conversational_chain(vector_store)
    clear_cache()
    
    total_docs_after = Document.query.count()
    stats_after = index.describe_index_stats()
    namespace_stats_after = stats_after.get('namespaces', {}).get(NAMESPACE, {})
    total_vec_after = namespace_stats_after.get('vector_count', 0)
    
    current_app.logger.info(
        f"[DEBUG][AFTER DELETE] File {file_id}: "
        f"MySQL docs={total_docs_after}, Pinecone vec total={total_vec_after}"
    )
    
    flash(
        f"✅ Data cleanup finished! MySQL documents: {total_docs_before} → {total_docs_after}, "
        f"Pinecone vectors: {total_vec_before} → {total_vec_after}.",
        "success"
    )

    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/delete_all_documents', methods=['POST'])
def delete_all_documents():
    Document.query.delete()
    db.session.commit()
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        grpc=True
    )
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception as e:
        current_app.logger.warning(f"Namespace '{NAMESPACE}' not found, skipping deletion: {e}")

    stats = index.describe_index_stats()
    namespace_stats = stats.get('namespaces', {}).get(NAMESPACE, {})
    remaining_vectors = namespace_stats.get('vector_count', 0) 
    total_docs = Document.query.count()
    current_app.logger.info(
        f"[STAT] After delete_all: MySQL docs={total_docs}, Pinecone vectors={remaining_vectors}"
    )
    flash(
        f"✅ Data has been cleaned! MySQL documents: {total_docs}, remaining items in Pinecone database: {remaining_vectors}",
        "success"
    )

    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({"history": conversation_history})