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
# PINECONE_SETTINGS = {
#     "INDEX_NAME": "cardiolytics",
#     "NAMESPACE": "cardiobot",
#     "SPEC": {"cloud": "aws", "region": "us-east-1"},
#     "EMBEDDING_DIMENSION": 1024
# }

PINECONE_SETTINGS = {
    "INDEX_NAME": "cardiolitics",
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
- Pahami maksud pertanyaan meskipun terdapat **kesalahan ketik (typo)**, ejaan tidak baku (misalnya, "bagaiman" berarti "bagaimana" atau "menceganhya" berarti "mencegahnya"), atau struktur kalimat yang tidak sempurna.
- Gunakan riwayat percakapan (terutama 3-5 pertanyaan terakhir di {chat_history}) untuk memahami konteks pengguna secara dinamis. Jika {chat_history} kosong atau tidak ada riwayat konteks yang jelas, **wajib anggap pertanyaan sebagai tidak spesifik** dan **larang sepenuhnya membuat asumsi tentang topik berdasarkan dokumen saja tanpa konteks eksplisit**. Dalam hal ini, tambahkan tag internal `[NO_INFO_TRIGGER]` di awal jawaban untuk keperluan sistem, tetapi **jangan tampilkan tag ini kepada pengguna**, dan berikan **hanya** respons bersih: "Maaf, saya tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan lebih spesifik! ✍️😊", tanpa menambahkan konten atau poin lain. **Contoh**: Untuk pertanyaan "bagaimana mencegahnya" tanpa riwayat, jawaban harus `[NO_INFO_TRIGGER] Maaf, saya tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan lebih spesifik! ✍️😊`, bukan konten rinci berdasarkan dokumen.
- **Penanganan Pertanyaan**:
  - Jika pertanyaan cukup spesifik dan terkait kardiovaskular (didasarkan pada riwayat konteks atau indikasi eksplisit seperti menyebutkan 'jantung' atau kondisi terkait) dan informasi tersedia dalam dokumen, berikan jawaban terperinci menggunakan format **Markdown** dengan heading dan list bernomor untuk poin atau langkah.
  - Jika pertanyaan meminta jawaban dalam bahasa lain (misalnya, "Can you speak in English?", "Can you speak with France?") atau tidak relevan dengan kardiovaskular (misalnya, "kapan indonesia merdeka"), tambahkan tag internal `[NO_SNIPPETS]` di awal jawaban untuk keperluan sistem, tetapi **jangan tampilkan tag ini kepada pengguna**. Berikan respons bersih sebagai berikut:
    - Untuk permintaan bahasa: "Maaf, saya hanya bisa menjawab dalam bahasa Indonesia sesuai topik kardiovaskular atau dokumen yang tersedia. Silakan ajukan pertanyaan ya! 😊"
    - Untuk pertanyaan non-kardiovaskular: "Maaf, saya tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan lebih spesifik! ✍️😊"
  - Pastikan teks yang dilihat pengguna tidak menyertakan `[NO_INFO_TRIGGER]`, `[NO_SNIPPETS]`, atau bagian teknis lainnya.
- **Pertanyaan Pancingan**:
  - Setelah jawaban terkait kardiovaskular yang berisi informasi spesifik dari dokumen, tambahkan satu pertanyaan pancingan (bersifat ya/tidak, relevan, dan memperdalam topik) yang **berdasarkan informasi dalam dokumen**. Contoh:
    - "Apakah Anda ingin tahu lebih lanjut tentang [subtopik spesifik, misalnya 'profil lipid']?"
    - "Ingin saya jelaskan lebih lanjut tentang [topik spesifik, misalnya 'tes darah untuk kolesterol']?"
    - "Perlu saya bantu jelaskan [aspek spesifik, misalnya 'faktor risiko yang dapat diubah']?"
  - Jangan tambahkan pertanyaan pancingan untuk respons yang mengandung tag internal `[NO_INFO_TRIGGER]` atau `[NO_SNIPPETS]`.

**PERTANYAAN:** {question}
**KONTEKS:** {context}
**RIWAYAT:** {chat_history}

**Jawaban:**
"""