from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import os
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import pdfplumber
import csv
import threading

app = Flask(__name__)
CORS(app)

# Define global variables
VECTORSTORE = None

# Split documents
MARKDOWN_SEPARATORS = ["\n#{1,6}", "\n\\\\\\*+\n", "\n---+\n", "\n_+\n", "\n\n", "\n", ". ", ""]

def split_documents(chunk_size, knowledge_base):
    text_splitter = RecursiveCharacterTextSplitter(
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

def load_pdf_as_documents(file_path):
    text = extract_text_from_pdf(file_path)
    return [LangchainDocument(page_content=text)]

# Create vectorstore using Ollama embeddings
def create_vectorstore(docs_processed):
    try:
        print("Initializing Ollama embeddings...")
        embeddings = OllamaEmbeddings(model="llama3")
        print("Embeddings initialized successfully.")
        
        print("Creating Chroma vectorstore...")
        vectorstore = Chroma.from_documents(documents=docs_processed, embedding=embeddings)
        print("Vectorstore created successfully.")
        
        return vectorstore
    except Exception as e:
        print(f"An error occurred during vectorstore creation: {e}")
        return None

# Function to call the Ollama LLM
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Function to perform RAG (Retrieve and Generate) operation
def rag_chain(question):
    retriever = VECTORSTORE.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)

# Summarizer function for text files
def summarizer(filepath):
    input_text = file_preprocessing(filepath)
    prompt = f"Summarize the following text:\n\n{input_text}\n\nSummary:"
    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={"max_tokens": 500, "temperature": 0.5}
        )

        # Since the response is a dictionary, directly access the 'response' key
        summary = response.get("response", "").strip()
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "An error occurred during summarization."

# Function to extract and format data from CSV
def extract_and_format_csv_data(file_path):
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Extract the headers
            rows = [row for row in reader]  # Extract the data rows

        # Format the data as a string
        data_string = "CSV Data:\n"
        data_string += ", ".join(headers) + "\n"
        data_string += "\n".join([", ".join(row) for row in rows])
        
        return data_string
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return ""

# Summarizer function for CSV files
def summarizer_csv(filepath):
    input_text = extract_and_format_csv_data(filepath)
    prompt = f"Summarize the following CSV data:\n\n{input_text}\n\nSummary:"
    
    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={"max_tokens": 500, "temperature": 0.5}
        )

        # Since the response is a dictionary, directly access the 'response' key
        summary = response.get("response", "").strip()
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "An error occurred during summarization."

def extract_tables_from_pdf(pdf_path, csv_path):
    try:
        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            csv_writer.writerow(row)
        print(f"Data has been written to {csv_path}")
        return True
    except Exception as e:
        print(f"Error extracting tables from PDF: {e}")
        return False

def load_dataset_and_process(file_path):
    try:
        # First extract tables from PDF and summarize CSV if applicable
        csv_path = file_path.replace('.pdf', '.csv')
        if file_path.endswith('.pdf') and extract_tables_from_pdf(file_path, csv_path):
            csv_summary = summarizer_csv(csv_path)
            text = extract_text_from_pdf(file_path) + "\n\n" + csv_summary
        else:
            text = extract_text_from_pdf(file_path) if file_path.endswith('.pdf') else open(file_path, 'r', encoding='utf-8').read()

        docs = [LangchainDocument(page_content=text)]

        print("Documents loaded successfully")
        
        print("Splitting documents...")
        docs_processed = split_documents(512, docs)
        print(f"Number of processed documents: {len(docs_processed)}")
        
        print("Creating vectorstore...")
        global VECTORSTORE
        VECTORSTORE = create_vectorstore(docs_processed)
        if VECTORSTORE:
            print("Vectorstore created successfully.")
        else:
            print("Failed to create vectorstore.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def file_preprocessing(file_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

@app.route('/upload', methods=['POST'])
def upload_file():
    global VECTORSTORE
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
        # Generate summary
        summary = summarizer(file_path)
        
        # Return the summary first
        response = jsonify({'message': 'File successfully uploaded', 'summary': summary})
        
        # After sending the response, process the dataset in a new thread
        threading.Thread(target=load_dataset_and_process, args=(file_path,)).start()
        
        return response, 200
    else:
        return jsonify({'error': 'Unsupported file format. Please upload a .pdf or .txt file.'}), 400

@app.route('/ask', methods=['POST'])
def ask():
    global VECTORSTORE
    if VECTORSTORE is None:
        return jsonify({'error': 'Vectorstore is not initialized'}), 500

    user_query = request.json.get('query')
    if user_query:
        answer = rag_chain(user_query)
        print("Generated Answer:", answer)
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'No query provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
