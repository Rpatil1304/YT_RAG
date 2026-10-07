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
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate


# Get available transcripts

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


# Select transcript

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


# Extract transcript

def extract_transcript(video_id):

    try:

        transcripts = get_available_transcripts(video_id)

        if not transcripts:

            print("No transcripts available.")

            return None, None

        print("\nAvailable transcripts:\n")

        for transcript in transcripts:

            print(
                f"{transcript['language']} "
                f"({transcript['language_code']}) "
                f"- Generated: {transcript['is_generated']}"
            )

        selected = select_transcript(transcripts)

        if selected is None:

            print("\nNo English or Hindi transcript found.")

            return None, None

        print("\nSelected transcript:")
        print(selected.language)

        fetched_transcript = selected.fetch()

        transcript_text = " ".join(
            snippet.text
            for snippet in fetched_transcript
        )

        return transcript_text, selected.language_code

    except TranscriptsDisabled:

        print("Transcripts are disabled for this video.")

        return None, None

    except NoTranscriptFound:

        print("No suitable transcript found.")

        return None, None

    except RequestBlocked:

        print("YouTube has blocked this request/IP.")
        print("Try running the script locally.")

        return None, None


# Load Hindi to English translation model

MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


# Translate Hindi to English

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


# Get English transcript

def get_english_transcript(video_id):

    transcript_text, language_code = extract_transcript(
        video_id
    )

    if transcript_text is None:

        return None

    if language_code == "en":

        print("\nEnglish transcript found.")

        return transcript_text

    elif language_code == "hi":

        print("\nHindi transcript found.")
        print("Translating Hindi → English...")

        english_text = translate_hindi_to_english(
            transcript_text
        )

        return english_text

    else:

        print(
            f"Unsupported language: {language_code}"
        )

        return None


# Main program

if __name__ == "__main__":

    # Video ID

    video_id = "d6mAi5kHsxc"


    # Get English transcript

    english_transcript = get_english_transcript(
        video_id
    )


    if english_transcript:

        # Text chunking

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

            print(f"--- Chunk {i} ---")
            print(chunk)
            print()


        # Create embeddings

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        # Create documents

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


        # Create FAISS vector store

        vector_store = FAISS.from_documents(
            documents,
            embedding_model
        )


        # Create retriever

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )

        # Load LLM

        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.2
        )


        # Create prompt

        prompt = PromptTemplate(
            template="""
        Answer the question using only the provided context.

        If the answer is not present in the context, say:
        "I don't know based on the provided video."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
            input_variables=["context", "question"]
        )
        # User query

        query = "What is the main topic of the video?"


        # Retrieve relevant documents

        retrieved_documents = retriever.invoke(
            query
        )


        # Display retrieved documents

        print("\n" + "=" * 60)
        print("RETRIEVED DOCUMENTS")
        print("=" * 60)


        for document in retrieved_documents:

            print(
                f"ID: {document.metadata['chunk_id']}"
            )

            print(
                f"Text: {document.page_content}"
            )

            print()
                # Create context

        context = "\n\n".join(
            document.page_content
            for document in retrieved_documents
        )


        # Create formatted prompt

        formatted_prompt = prompt.format(
            context=context,
            question=query
        )


        # Generate answer

        answer = llm.invoke(
            formatted_prompt
        )


        # Display answer

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(answer)