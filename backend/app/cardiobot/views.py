import os
import docx2txt
import pytz
import difflib
import re
import itertools
from io import BytesIO
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from app.extensions import db
from app.models import Document

load_dotenv()
cardiobot = Blueprint('cardiobot', __name__, url_prefix='/cardiobot')
groq_api_key = os.getenv('GROQ_API_KEY')
wib = pytz.timezone('Asia/Jakarta')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx', 'pptx', 'xlsx'}
CHROMA_PERSIST_DIR = "./chroma_db"

# Variabel global untuk sesi chatbot
conversation_history = []
vector_store = None
chain = None
has_greeted = False  # Flag untuk sapaan awal

# Daftar kata kunci
greeting_keywords = [
    "hi", "halo", "hai", "hey", "tes", "hello", "selamat pagi", 
    "selamat siang", "selamat sore", "selamat malam", "assalamualaikum", "wassalamualaikum"
]
relevant_keywords = [
    "kardiovaskular", "jantung", "pembuluh darah", "penyakit jantung", "sistem kardiovaskular", 
    "cardiovascular", "kardiovascular", "detak jantung", "kesehatan jantung", "stroke", 
    "kolesterol", "hipertensi", "tekanan darah", "serangan jantung"
]
general_keywords = [
    "bagaimana", "cara", "solusi", "penanganan", "pengobatan", "pencegahan", "treatment", "mengatasi", 
    "mengobati", "menghindari", "mencegah", "penyembuhan", "penyebab", "gejala", "resiko", 
    "diagnosis", "terapi", "obat", "perawatan", "langkah", "tindakan"
]
lifestyle_keywords = [
    "aktivitas fisik", "gaya hidup", "diet", "nutrisi", "vitamin", "olahraga", "stress", 
    "relaksasi", "kontrol tekanan darah", "pola makan", "pantangan", "perubahan gaya hidup",
    "pemeriksaan rutin", "cek kesehatan", "minuman beralkohol", "merokok", "kebiasaan makan", 
    "pola tidur", "diet rendah garam", "diet jantung"
]
medical_keywords = [
    "penyakit jantung", "hipertensi", "kolesterol tinggi", "gagal jantung", "serangan jantung", 
    "stroke", "aritmia", "kateterisasi", "angioplasti", "operasi jantung", "rehabilitasi", 
    "alat pacu jantung", "statin", "beta blocker", "ACE inhibitor", "antihipertensi", "antikoagulan"
]
diagnostic_keywords = [
    "tes darah", "elektrokardiogram", "ekokardiogram", "deteksi dini", "pemantauan tekanan", 
    "pemeriksaan", "kesehatan jantung", "detak jantung", "denyut nadi"
]
risk_keywords = [
    "faktor risiko", "komplikasi", "kondisi kronis", "kondisi akut", "bahaya merokok", 
    "akumulasi lemak", "inflamasi", "cedera pembuluh darah", "gula darah", "diabetes", "suplemen"
]


# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_mime_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    mime_types = {
        'pdf': 'application/pdf', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
        'png': 'image/png', 'txt': 'text/plain'
    }
    return mime_types.get(ext, 'application/octet-stream')

def load_or_initialize_vector_store():
    global vector_store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        vector_store = None
    return vector_store

def is_relevant_input(user_input):
    user_input_lower = user_input.lower()
    return any(difflib.SequenceMatcher(None, user_input_lower, keyword).ratio() > 0.5 for keyword in relevant_keywords)

def is_follow_up_question(user_input):
    follow_up_keywords = list(itertools.chain(general_keywords, lifestyle_keywords, medical_keywords, diagnostic_keywords, risk_keywords))
    return any(re.search(rf'\b{keyword}\b', user_input.lower()) for keyword in follow_up_keywords)


# Chatbot Functions
def create_conversational_chain(vector_store):
    chat_prompt = PromptTemplate(template='''
        Instruksi:
        - Anda adalah bot yang akan menjawab pertanyaan kardiovaskular berdasarkan dokumen yang telah diunggah.
        - Jawab hanya dalam bahasa Indonesia.
        - Jika pengguna meminta menggunakan bahasa Inggris, tolak permintaan dan tetap berbahasa Indonesia.
        - Jika pengguna menyapa di awal dan `has_greeted` belum `True`, balas dengan sapaan satu kali: "Halo! Saya adalah Cardiobot. Saya hanya akan menjawab pertanyaan seputar kardiovaskular." Lalu set `has_greeted` ke `True`.
        - Untuk pertanyaan berikutnya, langsung berikan jawaban tanpa mengulangi sapaan, fokus pada jawaban relevan sesuai konteks dokumen.
        - Pertahankan konteks dari pertanyaan terakhir. Jika pertanyaan lanjutan seperti "bagaimana cara penanganannya" diberikan, jawab dengan mengacu pada topik yang baru saja dibahas.
        - Jawab hanya pertanyaan yang relevan dengan dokumen ini. Abaikan atau tolak pertanyaan yang di luar konteks kardiovaskular.
        - Jika pertanyaan tidak dapat dijawab berdasarkan dokumen ini, mintalah pengguna untuk memberikan pertanyaan ulang dengan konteks yang lebih jelas.
        - Jika jawaban membutuhkan langkah-langkah, tuliskan setiap langkah di baris baru dengan nomor urutan yang jelas.
        
    **Pertanyaan Pengguna:** {question}  
    **Konteks dari Dokumen:** {context}
    **Konteks Percakapan Sebelumnya:** {chat_history}
    ''')

    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.2-3b-preview')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": chat_prompt,
            "document_variable_name": "context"
        }
    )

vector_store = load_or_initialize_vector_store()
if vector_store:
    chain = create_conversational_chain(vector_store)


@cardiobot.route('/cardiobot/chat', methods=['POST'])
def chat():
    global chain, vector_store, has_greeted, conversation_history

    if not vector_store or not chain:
        return jsonify({"error": "Dokumen belum diunggah atau database embedding tidak tersedia. Silakan unggah dokumen terlebih dahulu agar bot dapat memberikan jawaban."}), 400

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Pesan tidak valid. 'message' tidak boleh kosong."}), 400

    if user_input.lower() in greeting_keywords and not has_greeted:
        has_greeted = True
        return jsonify({"answer": "Halo! Saya adalah Cardiobot. Saya hanya akan menjawab pertanyaan seputar kardiovaskular."})

    if is_follow_up_question(user_input) and conversation_history:
        last_topic = conversation_history[-1]["user"]
        user_input = f"{last_topic} - {user_input}"

    try:
        result = chain.invoke({"question": user_input})
        answer = result.get("answer")

        if not answer:
            return jsonify({"error": "Dokumen tidak memiliki informasi yang relevan untuk menjawab pertanyaan Anda."}), 404

        conversation_history.append({"user": user_input, "bot": answer})
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": "Terjadi kesalahan di server."}), 500


# Document Functions
def reload_vector_store():
    global vector_store, chain
    documents = Document.query.all()
    all_text = ""

    for document in documents:
        filename = document.title_file
        file_data = document.file_data

        if filename.endswith(".pdf"):
            pdf_reader = PdfReader(BytesIO(file_data))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif filename.endswith(".txt"):
            text = file_data.decode("utf-8")
        elif filename.endswith(".docx"):
            text = docx2txt.process(BytesIO(file_data))
        else:
            continue

        all_text += text

    if all_text.strip():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(all_text)

        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR, collection_name="docs")
        chain = create_conversational_chain(vector_store)
    else:
        vector_store = None
        chain = None

@cardiobot.route('/cardiobot/documents', methods=['GET', 'POST'])
def documents():
    global vector_store, chain

    if request.method == 'POST':
        files = request.files.getlist('files[]')

        if not files or not all(allowed_file(file.filename) for file in files):
            flash('Some files are not allowed or no file uploaded!', 'error')
            return redirect(url_for('cardiobot.documents'))

        all_text = ""
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_data = file.read()

                # Menyimpan dokumen baru ke database
                new_document = Document(
                    title_file=filename, 
                    file_data=file_data, 
                    created_at=datetime.now(wib),
                    updated_at=datetime.now(wib)
                )
                db.session.add(new_document)

                # Menyusun teks berdasarkan jenis file
                if filename.endswith(".pdf"):
                    pdf_reader = PdfReader(BytesIO(file_data))
                    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                elif filename.endswith(".txt"):
                    text = file_data.decode("utf-8")
                elif filename.endswith(".docx"):
                    text = docx2txt.process(BytesIO(file_data))
                else:
                    flash(f"Tipe file tidak didukung: {filename}", 'error')
                    db.session.rollback()
                    return redirect(url_for('cardiobot.documents'))

                all_text += text

        db.session.commit()
        reload_vector_store()
        flash('Documents uploaded, indexed, and embedded in Chroma DB successfully!', 'success')
        return redirect(url_for('cardiobot.documents'))

    # Menampilkan dokumen dengan paginasi
    page = request.args.get('page', 1, type=int)
    per_page = 10
    search_query = request.args.get('search', '')

    if search_query:
        documents = Document.query.filter(Document.title_file.ilike(f'%{search_query}%')).paginate(page=page, per_page=per_page)
    else:
        documents = Document.query.paginate(page=page, per_page=per_page)

    return render_template('cardiobot/documents.html', navbar_title='Cardiobot/Documents', documents=documents, search_query=search_query)


@cardiobot.route('/cardiobot/live-chat')
def live_chat():
    return render_template('cardiobot/live_chat.html', navbar_title='Cardiobot')


@cardiobot.route('/cardiobot/documents/preview/<int:file_id>')
def preview_file(file_id):
    document = Document.query.get_or_404(file_id)
    return send_file(BytesIO(document.file_data), download_name=document.title_file, as_attachment=False)


@cardiobot.route('/cardiobot/documents/view/<int:file_id>')
def view_document(file_id):
    document = Document.query.get_or_404(file_id)
    mime_type = get_mime_type(document.title_file)
    preview_url = url_for('cardiobot.preview_file', file_id=document.id)
    
    return render_template(
        'cardiobot/view_document.html', navbar_title='Cardiobot/Documents/View',
        document=document, mime_type=mime_type, preview_url=preview_url
    )


@cardiobot.route('/cardiobot/documents/update/<int:file_id>', methods=['POST'])
def update_file(file_id):
    document = Document.query.get_or_404(file_id)
    new_file = request.files.get('new_file')

    if new_file and allowed_file(new_file.filename):
        filename = secure_filename(new_file.filename)
        file_data = new_file.read()

        document.title_file = filename
        document.file_data = file_data
        document.updated_at = datetime.now(wib)
        db.session.commit()

        reload_vector_store()
        flash('Document updated successfully!', 'success')
    else:
        flash('File not allowed or no file selected!', 'error')

    return redirect(url_for('cardiobot.documents'))


@cardiobot.route('/cardiobot/documents/download/<int:file_id>')
def download_file(file_id):
    document = Document.query.get_or_404(file_id)
    if document.file_data:
        return send_file(BytesIO(document.file_data), download_name=document.title_file, as_attachment=True)
    else:
        flash("No file data found!", "error")
        return redirect(url_for('cardiobot.documents'))


@cardiobot.route('/cardiobot/documents/delete/<int:file_id>', methods=['POST'])
def delete_document(file_id):
    document = Document.query.get_or_404(file_id)
    db.session.delete(document)
    db.session.commit()
    
    reload_vector_store()
    flash('Document deleted successfully!', 'success')
    return redirect(url_for('cardiobot.documents'))


@cardiobot.route('/cardiobot/delete_all_documents', methods=['POST'])
def delete_all_documents():
    try:
        Document.query.delete()
        db.session.commit()
        reload_vector_store()
        flash('All documents have been deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting documents: {str(e)}', 'error')
    
    return redirect(url_for('cardiobot.documents'))


@cardiobot.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({"history": conversation_history})
