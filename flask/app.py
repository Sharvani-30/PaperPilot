# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import fitz  # PyMuPDF
# import pyttsx3
# import os
# import  datasets
# from tqdm import tqdm
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores.utils import DistanceStrategy
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
# import torch
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import PyPDF2
# from datasets import Dataset

# app = Flask(__name__)
# CORS(app)

# # Define global variables
# KNOWLEDGE_VECTOR_DATABASE = None
# READER_LLM = None

# # Split documents
# MARKDOWN_SEPARATORS = ["\n#{1,6}", "\n\\\\\\*+\n", "\n---+\n", "\n_+\n", "\n\n", "\n", ". ", ""]

# def split_documents(chunk_size, knowledge_base, tokenizer_name="thenlper/gte-small"):
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name),
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=MARKDOWN_SEPARATORS,
#     )
#     docs_processed = []
#     for doc in knowledge_base:
#         docs_processed += text_splitter.split_documents([doc])
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)
    
#     return docs_processed_unique

# def extract_text_from_pdf(file_path):
#     with open(file_path, "rb") as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = []
#         for page in pdf_reader.pages:
#             text.append(page.extract_text())
#     return "\n".join(text)

# def load_pdf_as_dataset(file_path):
#     text = extract_text_from_pdf(file_path)
#     data = {"text": [text]}  # Wrap the text in a list to create a single example
#     dataset = Dataset.from_dict(data)
#     return dataset

# # Create knowledge vector database
# def create_knowledge_vector_database(docs_processed):
#     try:
#         print("Initializing embedding model...")
#         embedding_model = HuggingFaceEmbeddings(
#             model_name="thenlper/gte-small",
#             multi_process=True,
#             model_kwargs={"device": "cpu:0"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#         print("Embedding model initialized successfully.")
        
#         print("Creating FAISS index...")
#         knowledge_vector_database = FAISS.from_documents(
#             docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#         )
#         print("FAISS index created successfully.")
        
#         return knowledge_vector_database
#     except Exception as e:
#         print(f"An error occurred during knowledge vector database creation: {e}")
#         return None

# # Load model for answering queries
# def load_model():
#     global READER_LLM
#     READER_MODEL_NAME = "C://Users//RUYYADI SATHWIK//OneDrive//Desktop//Lamini//Lamini"
#     model = AutoModelForSeq2SeqLM.from_pretrained(READER_MODEL_NAME, torch_dtype=torch.float32, device_map='cpu')
#     tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
#     READER_LLM = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# def file_preprocessing(file_path: str) -> str:
#     if file_path.lower().endswith('.pdf'):
#         pdf_document = fitz.open(file_path)
#         text = ""
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)
#             text += page.get_text()
#         return text

#     elif file_path.lower().endswith('.txt'):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#         return content
#     else:
#         raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

# def summarizer(filepath):
#     tokenizer = T5Tokenizer.from_pretrained('t5-small')
#     base_model = T5ForConditionalGeneration.from_pretrained('t5-small')
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=5000,
#         min_length=50  
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result
# '''
# @app.route('/tts', methods=['POST'])
# def text_to_speech():
#     summary = request.json.get('summary')
#     if not summary:
#         return jsonify({"error": "No summary provided"})

#     # Initialize the TTS engine
#     engine = pyttsx3.init()

#     # Set properties (optional)
#     engine.setProperty('rate', 150)
#     engine.setProperty('volume', 1.0)

#     # Choose a voice (optional)
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[1].id)

#     # Convert summary to speech
#     engine.say(summary)
#     engine.runAndWait()

#     return jsonify({"message": "Text to speech completed"})
# '''
# def load_dataset_and_process(file_path):
#     try:

#         if file_path.endswith('.pdf'):
#             print(f"Loading dataset from PDF file: {file_path}")
#             ds = load_pdf_as_dataset(file_path)
#         else:
#             print(f"Loading dataset from text file: {file_path}")
#             ds = datasets.load_dataset("text", data_files=file_path, split="train", cache_dir="/cache")

#         print("Dataset loaded successfully")
        
#         print("Processing raw knowledge base...")
#         RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc["text"]) for doc in tqdm(ds)]
#         print(f"Number of documents in raw knowledge base: {len(RAW_KNOWLEDGE_BASE)}")
        
#         print("Splitting documents...")
#         docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE)
#         print(f"Number of processed documents: {len(docs_processed)}")
        
#         print("Creating knowledge vector database...")
#         global KNOWLEDGE_VECTOR_DATABASE
#         KNOWLEDGE_VECTOR_DATABASE = create_knowledge_vector_database(docs_processed)
#         if KNOWLEDGE_VECTOR_DATABASE:
#             print("Knowledge vector database created successfully.")
#         else:
#             print("Failed to create knowledge vector database.")
        
#     except UnicodeDecodeError as e:
#         # Handle the specific encoding issue here
#         print(f"UnicodeDecodeError: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global KNOWLEDGE_VECTOR_DATABASE
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
#         # Save the file
#         uploads_dir = os.path.abspath('uploads')
#         file_path = os.path.join(uploads_dir, file.filename)
#         file.save(file_path)

#         print(f"File saved to: {file_path}")

#         # Process the dataset using the uploaded file
#         load_dataset_and_process(file_path)

#         # Generate summary
#         summary = summarizer(file_path)
#         return jsonify({'message': 'File successfully uploaded', 'summary': summary}), 200
#     else:
#         return jsonify({'error': 'Unsupported file format. Please upload a .pdf or .txt file.'}), 400

# @app.route('/ask', methods=['POST'])
# def ask():
#     global READER_LLM, KNOWLEDGE_VECTOR_DATABASE
#     if READER_LLM is None:
#         load_model()

#     user_query = request.json.get('query')
#     if KNOWLEDGE_VECTOR_DATABASE:
#         print("Entered here")
#         retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
#         retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
#         context = "\nExtracted documents:\n" + "\n".join([f"Document {i}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

#         prompt_in_chat_format = [
#             {
#                 "role": "system",
#                 "content": """Using the information contained in the context,
# give a comprehensive answer to the question.
# Respond only to the question asked, response should be concise and relevant to the question.
# If the answer cannot be deduced from the context, do not give an answer.""",
#             },
#             {
#                 "role": "user",
#                 "content": f"""Context:

# {context}
# ---
# Now here is the question you need to answer.

# Question: {user_query}""",
#             },
#         ]

#         RAG_PROMPT_TEMPLATE = "\n".join(
#             [f"{part['role']}: {part['content']}" for part in prompt_in_chat_format]
#         )

#         final_prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context)
#         print("Final Prompt", final_prompt)
#         try:
#             answer = READER_LLM(final_prompt)[0]["generated_text"]
#             print("Answer", answer)
#             return jsonify({"answer": answer})
#         except Exception as e:
#             return jsonify({'error': f'Error generating answer: {str(e)}'}), 500
#     else:
#         return jsonify({'error': 'Knowledge database not initialized. Please upload a file first.'}), 400

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')

#     load_model()

#     app.run(debug=True, use_reloader=False)

from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
from datasets import Dataset
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
CORS(app)

# Define global variables
KNOWLEDGE_VECTOR_DATABASE = None
READER_LLM = None

# Split documents
MARKDOWN_SEPARATORS = ["\n#{1,6}", "\n\\\\\\*+\n", "\n---+\n", "\n_+\n", "\n\n", "\n", ". ", ""]

def split_documents(chunk_size, knowledge_base, tokenizer_name="thenlper/gte-small"):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    
    return docs_processed_unique

def extract_text_from_pdf(file_path):
    try:
        pdf_document = fitz.open(file_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def load_pdf_as_dataset(file_path):
    text = extract_text_from_pdf(file_path)
    data = {"text": [text]}  # Wrap the text in a list to create a single example
    dataset = Dataset.from_dict(data)
    return dataset

# Create knowledge vector database
def create_knowledge_vector_database(docs_processed):
    try:
        print("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            multi_process=True,
            model_kwargs={"device": "cpu:0"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedding model initialized successfully.")
        
        print("Creating FAISS index...")
        knowledge_vector_database = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        print("FAISS index created successfully.")
        
        return knowledge_vector_database
    except Exception as e:
        print(f"An error occurred during knowledge vector database creation: {e}")
        return None

# Load model for answering queries
def load_model():
    global READER_LLM
    READER_MODEL_NAME = "C://Users//RUYYADI SATHWIK//OneDrive//Desktop//Lamini//Lamini"
    model = AutoModelForSeq2SeqLM.from_pretrained(READER_MODEL_NAME, torch_dtype=torch.float32, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    READER_LLM = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

def file_preprocessing(file_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)

    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

def summarizer(filepath):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    base_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=5000,
        min_length=50  
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

def load_dataset_and_process(file_path):
    try:
        if file_path.endswith('.pdf'):
            print(f"Loading dataset from PDF file: {file_path}")
            ds = load_pdf_as_dataset(file_path)
        else:
            print(f"Loading dataset from text file: {file_path}")
            ds = datasets.load_dataset("text", data_files=file_path, split="train", cache_dir="/cache")

        print("Dataset loaded successfully")
        
        print("Processing raw knowledge base...")
        RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc["text"]) for doc in tqdm(ds)]
        print(f"Number of documents in raw knowledge base: {len(RAW_KNOWLEDGE_BASE)}")
        
        print("Splitting documents...")
        docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE)
        print(f"Number of processed documents: {len(docs_processed)}")
        
        print("Creating knowledge vector database...")
        global KNOWLEDGE_VECTOR_DATABASE
        KNOWLEDGE_VECTOR_DATABASE = create_knowledge_vector_database(docs_processed)
        if KNOWLEDGE_VECTOR_DATABASE:
            print("Knowledge vector database created successfully.")
        else:
            print("Failed to create knowledge vector database.")
        
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

@app.route('/upload', methods=['POST'])
def upload_file():
    global KNOWLEDGE_VECTOR_DATABASE
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        # Save the file
        uploads_dir = os.path.abspath('uploads')
        os.makedirs(uploads_dir, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        print(f"File saved to: {file_path}")

        # Process the dataset using the uploaded file
        load_dataset_and_process(file_path)

        # Generate summary
        summary = summarizer(file_path)
        return jsonify({'message': 'File successfully uploaded', 'summary': summary}), 200
    else:
        return jsonify({'error': 'Unsupported file format. Please upload a .pdf or .txt file.'}), 400

@app.route('/ask', methods=['POST'])
def ask():
    global READER_LLM, KNOWLEDGE_VECTOR_DATABASE
    if READER_LLM is None:
        load_model()

    user_query = request.json.get('query')
    if KNOWLEDGE_VECTOR_DATABASE:
        print("Entered here")
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(user_query)
        context = ' '.join([doc.page_content for doc in retrieved_docs])

        # Generating the final answer
        answer = READER_LLM(f"question: {user_query} context: {context}")[0]["generated_text"]
        print("Generated Answer:", answer)
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'Knowledge vector database is not initialized'}), 500

if __name__ == '__main__':
    app.run(debug=True)
