import os
import shutil
import docx2txt
import pytz
import difflib
import re
from io import BytesIO
from datetime import datetime
import json

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, send_file, jsonify, current_app,
    abort
)
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LangDocument
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec

from app.extensions import db
from app.models import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

cardiobot = Blueprint('cardiobot', __name__, url_prefix='/cardiobot')

# API keys
groq_api_key    = os.getenv('GROQ_API_KEY')
openai_api_key  = os.getenv('OPENAI_API_KEY')
gemini_api_key  = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env     = os.getenv('PINECONE_ENVIRONMENT')

# Timezone
wib = pytz.timezone('Asia/Jakarta')

# Pinecone index config
INDEX_NAME = "cardiolytics"
NAMESPACE = "cardiobot"  

# File settings
ALLOWED_EXTENSIONS  = {'pdf', 'docx', 'txt'}

# In-memory state
conversation_history: list[dict] = []
chat_cache: dict[str, dict] = {}
vector_store = None
chain = None

# Simple greeting keywords
greeting_keywords = [
    "hi", "halo", "hai", "hey", "tes", "hello",
    "selamat pagi", "selamat siang", "selamat sore",
    "selamat malam", "assalamualaikum"
]

# Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    spec = ServerlessSpec(cloud="aws", region="us-east-1")  # cocok dengan dashboard kamu

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  
            metric="cosine",
            spec=spec
        )
    return pc

init_pinecone()

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

def load_or_initialize_vector_store(app=None) -> LangchainPinecone | None:
    global vector_store

    if vector_store:
        return vector_store

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    if INDEX_NAME not in pc.list_indexes().names():
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=spec
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'}
    )

    vector_store = LangchainPinecone(
        index_name=INDEX_NAME,
        embedding=embeddings,       
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

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    idx = pc.Index(INDEX_NAME)
    try:
        idx.delete(delete_all=True, namespace=NAMESPACE)
    except Exception:
        current_app.logger.warning("[reload] Namespace clear failed, continuing.")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device':'cpu'})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n","\n","."," "])
    
    docs_for_index=[]
    for d in docs:
        data = d.file_data
        text = ''
        if d.title_file.lower().endswith('.pdf'):
            reader=PdfReader(BytesIO(data)); text=''.join(p.extract_text() or '' for p in reader.pages)
        elif d.title_file.lower().endswith('.docx'):
            text=docx2txt.process(BytesIO(data))
        else:
            text=data.decode('utf-8',errors='ignore')
        for i,chunk in enumerate(splitter.split_text(text)):
            docs_for_index.append(LangDocument(page_content=chunk,
                                              metadata={"source":d.title_file,
                                                        "vector_id":f"{d.id}-{i}"}))

    vector_store = LangchainPinecone(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
        text_key="page_content",
    )
    vector_store.add_documents(
        documents=docs_for_index,
        ids=[md.metadata['vector_id'] for md in docs_for_index]
    )
    current_app.logger.info(f"[reload] Indexed {len(docs_for_index)} chunks.")
    return True


def create_conversational_chain(vs: Pinecone) -> ConversationalRetrievalChain:
    chat_prompt = PromptTemplate(template=
        """
        INSTRUKSI UNTUK LLM:
        - Jawab hanya berdasar dokumen yang diunggah.
        - Sertakan sumber dan nomor halaman.
        - Jika tidak ada, jawab: "Maaf, informasi tidak ditemukan."
        - Jawab dalam bahasa Indonesia.
        - **Gunakan format Markdown:**  
            • Heading untuk judul  
            • List bernomor untuk langkah-langkah  

        **Pertanyaan Pengguna:** {question}
        **Konteks Dokumen:** {context}
        **Riwayat Percakapan:** {chat_history}
        """
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=gemini_api_key,
        temperature=0.1
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 50}
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

# Initialize on startup
def initialize(state):
    global vector_store, chain
    app = state.app
    vs = load_or_initialize_vector_store(app)
    if not vs:
        app.logger.warning("⚠️ Pinecone index belum ada.")
        return
    
    chain = create_conversational_chain(vs)
    app.logger.info("✅ Cardiobot ready: vector_store & chain initialized.")

@cardiobot.route('/clear_cache', methods=['POST'])
def clear_cache():
    global chat_cache
    chat_cache.clear()
    current_app.logger.info("✅ [DEBUG] chat_cache cleared")
    return jsonify({"status": "ok"}), 200

@cardiobot.route('/chat', methods=['POST'])
def chat():
    global chat_cache, chain, vector_store, conversation_history

    data = request.get_json(silent=True) or {}

    # — NORMALISASI INPUT —
    raw = data.get("message", "") or ""
    # collapse semua whitespace jadi satu spasi, trim ujung, lalu lowercase
    user_input = re.sub(r"\s+", " ", raw).strip().lower()
    # — END NORMALISASI —

    current_app.logger.debug(f"[CHAT] Received message: {user_input!r}")

    # 1) Pastikan bot siap dan input tidak kosong
    if not vector_store or not chain or not user_input:
        current_app.logger.warning("[CHAT] Bot not ready or empty input")
        return jsonify({"error": "Bot belum siap atau input kosong."}), 400

    # 2) Cek cache
    if user_input in chat_cache:
        current_app.logger.debug(f"[CHAT] Cache HIT for key {user_input!r}")
        response = chat_cache[user_input]
        current_app.logger.debug(f"[CHAT] Returning from cache: {json.dumps(response)}")
        return jsonify(response)

    # 3) Greeting sederhana
    if user_input in greeting_keywords:
        resp = {
            "answer": "Halo! Saya adalah Cardiobot. Saya hanya akan menjawab pertanyaan seputar kardiovaskular.",
            "sources": []
        }
        chat_cache[user_input] = resp
        current_app.logger.debug(f"[CHAT] Greeting stored to cache under key {user_input!r}")
        return jsonify(resp)

    # 4) Cache miss → panggil chain
    try:
        res = chain.invoke({"question": user_input})
        answer = res["answer"]
        docs = res.get("source_documents", [])
        snippets = [
            {"file": d.metadata.get("source", "Unknown"), "text": s}
            for d in docs
            for s in (extract_sentences(answer, d.page_content) or [d.page_content[:200]])
        ]
        out = {"answer": answer, "sources": snippets}

        # simpan ke cache
        chat_cache[user_input] = out
        current_app.logger.debug(f"[CHAT] Cache MISS: stored response under key {user_input!r}")
        current_app.logger.debug(f"[CHAT] Current cache keys: {list(chat_cache.keys())}")

        conversation_history.append({"user": user_input, "bot": answer})
        return jsonify(out)

    except Exception as e:
        current_app.logger.error("Error di /chat: %s", e, exc_info=True)
        return jsonify({"error": "Terjadi kesalahan di server."}), 500


@cardiobot.route('/live-chat', defaults={'file_id': None})
@cardiobot.route('/live-chat/<int:file_id>')
def live_chat(file_id):
    
    if file_id is None:
        document = Document.query.first()
        if not document:
            abort(404, "Dokumen tidak ditemukan.")
    else:
        document = Document.query.get_or_404(file_id)

    return render_template(
        'cardiobot/live_chat.html',
        document=document,
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
    doc  = Document.query.get_or_404(file_id)
    mime = doc.title_file.rsplit('.',1)[1].lower()
    # Bangun URL untuk preview PDF
    preview_url = url_for('cardiobot.preview_file', file_id=file_id)
    return render_template(
        'cardiobot/view_document.html',
        navbar_title='Cardiobot/Documents View',
        document=doc,
        mime_type=f"application/{mime}",
        preview_url=preview_url,      # ← kirimkan di sini
    )


@cardiobot.route('/documents', methods=['POST'])
def upload_documents():
    global vector_store, chain      
    files = request.files.getlist('files[]')
    if not files or not all(allowed_file(f.filename) for f in files):
        flash("❌ File tidak valid atau tidak ada file.", "error")
        return redirect(url_for('admin.settings', tab='docs'))

    # Simpan ke MySQL
    for f in files:
        name = secure_filename(f.filename)
        data = f.read()
        db.session.add(Document(
            title_file=name,
            file_data=data,
            created_at=datetime.now(wib),
            updated_at=datetime.now(wib)
        ))
    db.session.commit()

    # ==== REBUILD PINECONE HANYA DI SINI ====
    success = reload_vector_store()
    if success:
        global chain
        chain = create_conversational_chain(vector_store)
    # =======================================

    clear_cache()
    
    total_docs = Document.query.count()
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    stats = pc.Index(INDEX_NAME).describe_index_stats(namespace=NAMESPACE)
    total_vectors = stats.total_vector_count

    current_app.logger.info(f"[DEBUG] MySQL docs={total_docs}, Pinecone vectors={total_vectors}")

    flash(
        f"✅ Indexed! (DB:{total_docs} docs, PC:{total_vectors} vectors)",
        "success"
    )
    
    
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/documents/delete/<int:file_id>', methods=['POST'])
def delete_document(file_id):
    # Inisialisasi client Pinecone
    pc    = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    index = pc.Index(INDEX_NAME)

    # 0) Debug awal: jumlah dokumen & vektor sebelum delete
    total_docs_before = Document.query.count()
    stats_before      = index.describe_index_stats(namespace=NAMESPACE)
    total_vec_before  = stats_before.total_vector_count
    current_app.logger.info(
        f"[DEBUG][BEFORE DELETE] File {file_id}: "
        f"MySQL docs={total_docs_before}, Pinecone vec total={total_vec_before}"
    )

    # 1) Ambil record & simpan blob-nya untuk chunking
    doc = Document.query.get_or_404(file_id)
    raw = doc.file_data

    # 2) Hapus record MySQL
    db.session.delete(doc)
    db.session.commit()

    # 3) Rekonstruksi teks & hitung jumlah chunk (vector_id)
    if doc.title_file.lower().endswith('.pdf'):
        reader = PdfReader(BytesIO(raw))
        text   = "".join(page.extract_text() or "" for page in reader.pages)
    elif doc.title_file.lower().endswith('.docx'):
        text   = docx2txt.process(BytesIO(raw))
    else:
        text   = raw.decode('utf-8', errors='ignore')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    n_chunks   = len(splitter.split_text(text))

    # 4) Hapus vector-file yang sesuai
    vector_ids = [f"{file_id}-{i}" for i in range(n_chunks)]
    try:
        index.delete(ids=vector_ids, namespace=NAMESPACE)
    except Exception as e:
        current_app.logger.warning(f"[DEBUG] Gagal delete vectors for doc {file_id}: {e}")

    # 5) REBUILD IN-MEMORY VECTOR_STORE & CHAIN
    #    agar vector_store & chain langsung sinkron tanpa menunggu restart
    reload_vector_store()
    global chain
    chain = create_conversational_chain(vector_store)

    # 6) CLEAR CHAT CACHE (opsional, supaya response lama tidak dipakai lagi)
    clear_cache()
    
    # 7) Debug setelah delete
    total_docs_after = Document.query.count()
    stats_after      = index.describe_index_stats(namespace=NAMESPACE)
    total_vec_after  = stats_after.total_vector_count
    current_app.logger.info(
        f"[DEBUG][AFTER DELETE] File {file_id}: "
        f"MySQL docs={total_docs_after}, Pinecone vec total={total_vec_after}"
    )

    flash(
        f"✅ Dokumen {file_id} di-DB MySQL: {total_docs_before}→{total_docs_after}; "
        f"Vektor Pinecone: {total_vec_before}→{total_vec_after}.",
        "success"
    )
    return redirect(url_for('admin.settings', tab='docs'))

@cardiobot.route('/delete_all_documents', methods=['POST'])
def delete_all_documents():
    Document.query.delete()
    db.session.commit()

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception:
        current_app.logger.info(f"Namespace '{NAMESPACE}' belum ada, skip delete_all.")

    # Debug: cek stats Pinecone
    stats = index.describe_index_stats(namespace=NAMESPACE)
    remaining_vectors = stats.total_vector_count  
    total_docs = Document.query.count()

    current_app.logger.info(
        f"[DEBUG] After delete_all: MySQL docs={total_docs}, Pinecone vectors={remaining_vectors}"
    )
    flash(
        f"✅ All cleared! MySQL docs={total_docs}; Pinecone vectors={remaining_vectors}",
        "success"
    )
    return redirect(url_for('admin.settings', tab='docs'))


@cardiobot.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({"history": conversation_history})


