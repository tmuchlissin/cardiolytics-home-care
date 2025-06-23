import pytz
import re
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

load_dotenv()

# API keys
GROQ_API_KEY = Config.GROQ_API_KEY
OPENAI_API_KEY = Config.OPENAI_API_KEY
GEMINI_API_KEY = Config.GEMINI_API_KEY
PINECONE_API_KEY = Config.PINECONE_API_KEY
PINECONE_ENV = Config.PINECONE_ENV

# Pinecone settings
PINECONE_SETTINGS = {
    "INDEX_NAME": "cardiolytics",
    "NAMESPACE": "cardiobot",
    "SPEC": {"cloud": "aws", "region": "us-east-1"},
    "EMBEDDING_DIMENSION": 1024
}

INDEX_NAME = PINECONE_SETTINGS["INDEX_NAME"]
NAMESPACE = PINECONE_SETTINGS["NAMESPACE"]
PINECONE_SPEC = PINECONE_SETTINGS["SPEC"]
EMBEDDING_DIMENSION = PINECONE_SETTINGS["EMBEDDING_DIMENSION"]

# Embedder
EMBEDDER = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

# File settings
FILE_SETTINGS = {
    "ALLOWED_EXTENSIONS": {'pdf'}
}
ALLOWED_EXTENSIONS = FILE_SETTINGS["ALLOWED_EXTENSIONS"]

# Timezone
TIMEZONE = pytz.timezone('Asia/Jakarta')

# Conversation context
CONVERSATION_CONTEXT = {
    "last_topic": None
}

# Chat settings
GREETING_KEYWORDS = [
    "hi", "halo", "hai", "hey", "tes", "hello",
    "selamat pagi", "selamat siang", "selamat sore",
    "selamat malam", "assalamualaikum"
]

FOLLOWUP_POSITIVE = {
    "ya", "iya", "betul", "benar", "setuju", "ya dong", "iya banget", "yoi", "sip",
    "lanjut", "oke lanjut", "lanjut saja", "silakan", "ya kak", "ya benar", "benar sekali",
    "iya bagaimana", "ya bagaimana", "iya apa saja", "ya apa saja", "iya sebutkan apa saja",
    "ya sebutkan apa saja", "iya apa saja sebutkan", "ya apa saja sebutkan", "teruskan",
    "boleh lanjut", "saya mau lanjut", "ya silakan", "iya lanjutkan", "oke dong",
    "setuju lanjut", "iya ayo", "ya ayo", "lanjut ya", "saya tertarik", "terus saja",
    "ya tentu", "iya tentu", "boleh dong", "ya mari", "iya mari", "lanjut bro",
    "oke mari", "ya jelaskan", "iya jelaskan", "lanjut kak", "saya setuju lanjut",
    "ya bagus", "iya bagus", "teruskan ya", "boleh terangkan", "ya terangkan",
    "iya terangkan", "lanjutkan saja", "ya oke", "iya oke"
}

FOLLOWUP_NEGATIVE = {
    "tidak", "tidak perlu", "tidak kok", "tidak lanjut", "tidak mau", "tidak usah", "ga usah",
    "nggak usah", "jangan", "cukup", "udah", "skip", "tidak ingin"
}

FOLLOWUP_NEUTRAL = {
    "kurang paham", "bingung", "tidak tahu", "kurang jelas", "masih bingung", "tidak yakin",
    "belum jelas", "tidak ngerti", "maksudnya?", "bisa diulang?", "gimana maksudnya",
    "ga ngerti", "nggak ngerti", "kurang ngerti", "coba jelaskan lagi", "ulangi", "jelaskan ulang",
    "saya tidak yakin", "tolong ulangi"
}

GENERAL_QUERY_PATTERN = re.compile(
    r"""(?x)
    \b(
        apa(\s+saja|\s+aja)? |
        adakah | apakah |
        bisakah | bolehkah |
        informasi | dokumen |
        tahu | ketahui |
        ada | punya |
        tersedia |
        kamu\s+(punya|tahu|miliki)
    )\b
    [\s\w,]{0,30}?
    \b(
        informasi | dokumen | file | data | isi |
        materi | topik | pengetahuan |
        yang\s+ada | yang\s+tersedia |
        bisa\s+diakses | dapat\s+dilihat |
        referensi | bacaan | daftar\s+materi |
        kumpulan\s+(file|data|topik) |
        dokumen\s+apa\s+saja | tersedia\s+apa
    )\b
    """,
    flags=re.IGNORECASE
)

CHAT_PROMPT_TEMPLATE = """
**INSTRUKSI:**
- Jawab **hanya berdasarkan konteks dokumen** menggunakan **bahasa Indonesia yang sopan, jelas, dan mudah dipahami oleh pasien awam**.
- Pahami maksud pertanyaan meskipun terdapat **kesalahan ketik (typo)**, ejaan tidak baku, atau struktur kalimat yang tidak sempurna.
- Gunakan riwayat percakapan (terutama 3-5 pertanyaan terakhir) untuk memahami konteks pengguna secara dinamis.
- Jika informasi tidak relevan atau tidak ditemukan, jawab: "Maaf, informasi tidak ditemukan dalam dokumen yang tersedia."
- Gunakan format **Markdown** dengan heading dan list bernomor untuk poin atau langkah.
- Setelah jawaban, tambahkan satu pertanyaan pancingan kepada pengguna (bersifat ya/tidak, relevan, dan memperdalam topik — bukan mengulang). Jangan berikan pertanyaan jika jawabannya "informasi tidak ditemukan".
    - ❗ **Pertanyaan harus relevan langsung dengan informasi yang baru Anda jelaskan dalam dokumen** dan harus dapat dijawab berdasarkan informasi dalam dokumen.
    - ❓ **Bentuk pertanyaan sebagai ajakan atau pilihan** yang spesifik, seperti:
        - "Apakah Anda ingin tahu lebih lanjut tentang [subtopik spesifik dari dokumen, misalnya 'profil lipid' atau 'pencegahan']?"
        - "Ingin saya jelaskan lebih lanjut tentang [topik spesifik, misalnya 'tes darah untuk kolesterol' atau 'pengobatan dengan statin']?"
        - "Perlu saya bantu jelaskan [aspek spesifik, misalnya 'faktor risiko yang dapat diubah']?"

**PERTANYAAN:** {question}
**KONTEKS:** {context}
**RIWAYAT:** {chat_history}
"""