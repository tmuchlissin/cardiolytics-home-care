import os
import pytz
from dotenv import load_dotenv
import re
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')

# Pinecone settings
INDEX_NAME = "cardiolytics"
NAMESPACE = "cardiobot"
PINECONE_SPEC = {"cloud": "aws", "region": "us-east-1"}
EMBEDDING_DIMENSION = 1024
EMBEDDER= HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'}
)

# File settings
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Timezone
TIMEZONE = pytz.timezone('Asia/Jakarta')

# Chat settings
GREETING_KEYWORDS = [
    "hi", "halo", "hai", "hey", "tes", "hello",
    "selamat pagi", "selamat siang", "selamat sore",
    "selamat malam", "assalamualaikum"
]

FOLLOWUP_POSITIVE = {
    "ya", "iya", "betul", "benar", "setuju", "ya dong", "iya banget", "yoi", "sip",
    "lanjut", "oke lanjut", "lanjut saja", "silakan", "ya kak", "ya benar", "benar sekali",
    'iya bagaimana', 'ya bagaimana', "iya apa saja", "ya apa saja", "iya sebutkan apa saja", 
    "ya sebutkan apa saja", "iya apa saja sebutkan", "ya apa saja sebutkan", "teruskan", 
    "boleh lanjut", "saya mau lanjut", "ya silakan", "iya lanjutkan", "oke dong",
    "setuju lanjut", "iya ayo", "ya ayo", "lanjut ya", "saya tertarik", "terus saja",
    "ya tentu", "iya tentu", "boleh dong", "ya mari", "iya mari", "lanjut bro",
    "oke mari", "ya jelaskan", "iya jelaskan", "lanjut kak", "saya setuju lanjut",
    "ya bagus", "iya bagus", "teruskan ya", "boleh terangkan", "ya terangkan",
    "iya terangkan", "lanjutkan saja", "ya oke", "iya oke"
}

FOLLOWUP_NEGATIVE = {
    "tidak", "tidak perlu", "tidak kok", "tidak lanjut", "tidak mau", "tidak usah", "ga usah", "nggak usah",
    "jangan", "cukup", "udah", "skip", "tidak ingin"
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
        informasi | dokumen | file | data | 
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
INSTRUKSI UNTUK LLM (WAJIB DIIKUTI):

1. Jawablah **hanya berdasarkan informasi yang terdapat dalam bagian "Konteks Dokumen"**.
2. **JANGAN** menambahkan informasi dari luar dokumen, asumsi, atau pendapat pribadi.
3. Jika tidak ada informasi yang relevan secara spesifik dalam dokumen untuk menjawab pertanyaan, atau jika dokumen yang tersedia tidak cukup relevan dengan topik pertanyaan (misalnya, hanya mengandung kata kunci umum tanpa menjelaskan topik spesifik), jawab dengan sopan: "Maaf, informasi tidak ditemukan dalam dokumen yang tersedia."
4. Gunakan **bahasa Indonesia** yang **sopan**, **jelas**, dan **mudah dipahami oleh pasien awam**.
5. Gunakan format **Markdown**:
    - Berikan heading untuk judul utama.
    - Gunakan list bernomor (`1.`, `2.`, dst.) untuk menjelaskan poin atau langkah.
6. Bila memungkinkan, **kutip langsung kalimat dari dokumen** sebagai bukti atau penegas jawaban, dan sebutkan sumbernya (misalnya, nama file dokumen).
7. Perlakukan kata "Anda" sebagai merujuk pada **pengguna manusia** yang sedang bertanya.
8. Setelah memberikan jawaban utama, **ajukan SATU pertanyaan lanjutan**:
    - ❗ **Pertanyaan harus relevan langsung dengan informasi yang baru Anda jelaskan dalam dokumen** dan harus dapat dijawab berdasarkan informasi dalam dokumen.
    - ❓ **Bentuk pertanyaan sebagai ajakan atau pilihan** yang spesifik, seperti:
        - "Apakah Anda ingin tahu lebih lanjut tentang [subtopik spesifik dari dokumen, misalnya 'profil lipid' atau 'pencegahan']?"
        - "Ingin saya jelaskan lebih lanjut tentang [topik spesifik, misalnya 'tes darah untuk kolesterol' atau 'pengobatan dengan statin']?"
        - "Perlu saya bantu jelaskan [aspek spesifik, misalnya 'faktor risiko yang dapat diubah']?"
    - 🟢 Pertanyaan ini **harus bisa dijawab dengan 'ya', 'tidak', atau variasinya** (misalnya, "tentu", "tidak perlu", "kurang jelas").
    - 🟡 Jika pengguna menjawab "ya" untuk pertanyaan lanjutan sebelumnya, **rujuk kembali ke riwayat percakapan atau konteks dokumen** untuk memilih satu subtopik spesifik (misalnya, gejala, pengobatan, atau faktor risiko yang disebutkan sebelumnya) dan ajukan pertanyaan lanjutan berdasarkan itu.
    - 🚫 Hindari pertanyaan terbuka seperti "mengapa?" atau "menurut Anda bagaimana?".
    - 🚫 Jangan ajukan pertanyaan lanjutan yang tidak dapat dijawab berdasarkan dokumen.
    - Berikan jarak newline "\n" sebelum pertanyaan lanjutan
9. Jika pengguna menjawab "ya" untuk pertanyaan lanjutan sebelumnya, **pilih satu subtopik spesifik** dari konteks dokumen atau riwayat percakapan (misalnya, 'profil lipid' atau 'tes darah' jika konteksnya tentang faktor risiko) dan berikan penjelasan lebih lanjut berdasarkan dokumen. Jika tidak ada subtopik yang jelas, pilih subtopik yang paling relevan dari dokumen.
10. Jangan berimajinasi atau menyusun jawaban tambahan jika tidak ada informasi dalam dokumen.
11. Jawaban Anda **harus memiliki kesamaan substansi yang kuat dengan dokumen** dan secara spesifik menjawab pertanyaan pengguna. Jika dokumen hanya mengandung informasi umum tanpa menjelaskan topik spesifik yang ditanyakan, kembalikan pesan: "Maaf, informasi tidak ditemukan dalam dokumen yang tersedia."
12. Jika jawaban adalah "Maaf, informasi tidak ditemukan dalam dokumen yang tersedia," **jangan ajukan pertanyaan lanjutan**.
13. Jika pertanyaan pengguna bersifat umum seperti "apa informasi yang Anda miliki?", berikan ringkasan singkat tentang topik utama yang dibahas dalam dokumen, seperti "Dokumen yang tersedia berisi informasi tentang dislipidemia, faktor risiko kardiovaskular, dan pengobatan jantung."

---

**PERTANYAAN PENGGUNA:**  
{question}

**KONTEKS DOKUMEN:**  
{context}

**RIWAYAT PERCAKAPAN SEBELUMNYA:**  
{chat_history}
"""

CONVERSATION_CONTEXT = {
    "last_topic": None  
}