from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
# 1. GET AVAILABLE TRANSCRIPTS
# ============================================================

def get_available_transcripts(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.list(video_id)

    transcripts = []

    for transcript in transcript_list:

        transcripts.append({
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "transcript_object": transcript
        })

    return transcripts


# ============================================================
# 2. SELECT TRANSCRIPT
# ============================================================

def select_transcript(transcripts):

    # First preference: English
    for transcript in transcripts:

        if transcript["language_code"] == "en":
            return transcript["transcript_object"]

    # Second preference: Hindi
    for transcript in transcripts:

        if transcript["language_code"] == "hi":
            return transcript["transcript_object"]

    return None


# ============================================================
# 3. EXTRACT TRANSCRIPT
# ============================================================

def extract_transcript(video_id):

    try:

        transcripts = get_available_transcripts(video_id)

        if not transcripts:

            print("No transcripts available.")

            return None, None


        # ----------------------------------------------------
        # Show available transcripts
        # ----------------------------------------------------

        print("\nAvailable transcripts:\n")

        for transcript in transcripts:

            print(
                f"{transcript['language']} "
                f"({transcript['language_code']}) "
                f"- Generated: {transcript['is_generated']}"
            )


        # ----------------------------------------------------
        # Select English or Hindi transcript
        # ----------------------------------------------------

        selected = select_transcript(transcripts)

        if selected is None:

            print(
                "\nNo English or Hindi transcript found."
            )

            return None, None


        print("\nSelected transcript:")
        print(selected.language)


        # ----------------------------------------------------
        # Fetch transcript
        # ----------------------------------------------------

        fetched_transcript = selected.fetch()


        # ----------------------------------------------------
        # Convert transcript into one text string
        # ----------------------------------------------------

        transcript_text = " ".join(
            snippet.text
            for snippet in fetched_transcript
        )


        return transcript_text, selected.language_code


    except TranscriptsDisabled:

        print(
            "Transcripts are disabled for this video."
        )

        return None, None


    except NoTranscriptFound:

        print(
            "No suitable transcript found."
        )

        return None, None


    except RequestBlocked:

        print(
            "YouTube has blocked this request/IP."
        )

        print(
            "Try running the script locally."
        )

        return None, None


# ============================================================
# 4. LOAD HINDI → ENGLISH TRANSLATION MODEL
# ============================================================

MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 5. TRANSLATE HINDI → ENGLISH
# ============================================================

def translate_hindi_to_english(hindi_text):

    inputs = tokenizer(
        hindi_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )


    translated_tokens = model.generate(
        **inputs
    )


    english_text = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )


    return english_text


# ============================================================
# 6. GET ENGLISH TRANSCRIPT
# ============================================================

def get_english_transcript(video_id):

    transcript_text, language_code = extract_transcript(
        video_id
    )


    if transcript_text is None:

        return None


    # --------------------------------------------------------
    # English transcript
    # --------------------------------------------------------

    if language_code == "en":

        print(
            "\nEnglish transcript found."
        )

        return transcript_text


    # --------------------------------------------------------
    # Hindi transcript
    # --------------------------------------------------------

    elif language_code == "hi":

        print(
            "\nHindi transcript found."
        )

        print(
            "Translating Hindi → English..."
        )


        english_text = translate_hindi_to_english(
            transcript_text
        )


        return english_text


    # --------------------------------------------------------
    # Unsupported language
    # --------------------------------------------------------

    else:

        print(
            f"Unsupported language: {language_code}"
        )

        return None



# ============================================================
# 7. MAIN PROGRAM
# ============================================================

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # VIDEO ID
    # Change ONLY this value for another video
    # --------------------------------------------------------

    video_id = "d6mAi5kHsxc"


    # --------------------------------------------------------
    # 1. Get English transcript
    # --------------------------------------------------------

    english_transcript = get_english_transcript(
        video_id
    )


    if english_transcript:

        # ====================================================
        # 2. TEXT CHUNKING
        # ====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(
            english_transcript
        )


        print("\n" + "=" * 60)
        print("TRANSCRIPT CHUNKS")
        print("=" * 60)

        print(
            f"Total chunks: {len(chunks)}\n"
        )


        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            print(
                f"--- Chunk {i} ---"
            )

            print(chunk)

            print()


        # ====================================================
        # 3. CREATE HUGGING FACE EMBEDDINGS
        # ====================================================

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        # ====================================================
        # 4. CREATE DOCUMENTS WITH CHUNK IDs
        # ====================================================

        documents = []

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            document = Document(
                page_content=chunk,

                metadata={
                    "chunk_id": f"chunk_{i}"
                }
            )

            documents.append(document)


        # ====================================================
        # 5. CREATE FAISS VECTOR STORE
        # ====================================================

        vector_store = FAISS.from_documents(
            documents,
            embedding_model
        )


        # ====================================================
        # 6. DISPLAY VECTOR STORE DOCUMENTS
        # ====================================================

        print("\n" + "=" * 60)
        print("VECTOR STORE DOCUMENTS")
        print("=" * 60)


        for document in documents:

            print(
                f"ID: {document.metadata['chunk_id']}"
            )

            print(
                f"Text: {document.page_content[:100]}..."
            )

            print()