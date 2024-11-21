
# PDF Document Query System Using FAISS and Google Generative AI
This project implements a PDF document query system that allows users to upload PDF files, extract their text, index it using FAISS (Fast Approximate Nearest Neighbor Search), and use Google Generative AI to answer user queries based on the contents of those documents.

## Features
- Upload PDF files to the system
- Text extraction from PDFs
- Text chunking and FAISS-based indexing
- Use of Google Generative AI for answering queries
- Query processing and document retrieval

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-query-system.git
   ```

2. **Create a virtual environment:**
conda create -p venv python==3.11.9 -y
   ```

3. **Activate the virtual environment:**
conda activate venv/

4. **Install dependencies:**
   pip install -r requirements.txt

5. **Set up environment variables:**
   - Create a `.env` file in the root directory of the project.
   - Add the following line to the `.env` file:
     GOOGLE_API_KEY=your_google_api_key

6. **Run the application:**
   uvicorn app:app --port 8002

7. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:8002/docs`.

8. **Optional: Install FAISS:**
   You may need to install the FAISS library:
   pip install faiss-cpu


## Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for using Google Generative AI.

## Usage
Once the system is up and running, you can interact with it via the following API endpoints:

### `/upload` - Upload PDFs
**Method:** POST  
**Parameters:**  
- `files` (multipart form data) - The PDFs to be uploaded.

### `/chat` - Ask a question about the uploaded documents
**Method:** POST  
**Parameters:**  
- `question` (form data) - The question you want to ask.

### Example:
- Upload a PDF file:
  ```bash
  curl -X 'POST'     'http://localhost:8000/upload'     -F 'files=@yourfile.pdf'
  ```
- Ask a question:
  ```bash
  curl -X 'POST'     'http://localhost:8000/chat'     -F 'question=What is the main topic of the document?'
  ```

The system will return an answer and the relevant sources (PDF filenames) used to generate the answer.

---

# API Documentation

## `/upload`
- **Method:** POST
- **Description:** Upload PDF files to the system for processing.
- **Request Body:**
  - `files`: One or more PDF files (multipart form data).
- **Response:**
  - **200 OK**: If files were uploaded successfully.
  - **400 Bad Request**: If no files are uploaded or there's an error.

### Example Request:
```bash
curl -X 'POST'   'http://localhost:8000/upload'   -F 'files=@document.pdf'
```

## `/chat`
- **Method:** POST
- **Description:** Ask a question about the uploaded PDF documents.
- **Request Body:**
  - `question`: The query you want to ask (form data).
- **Response:**
  - **200 OK**: If the query is processed successfully, returns a JSON with the answer and sources.
  - **400 Bad Request**: If the query fails or documents are not uploaded.
  - **500 Internal Server Error**: If there’s an issue during query processing.

### Example Request:
```bash
curl -X 'POST'   'http://localhost:8000/chat'   -F 'question=What is the main topic of the document?'
```

### Example Response:
```json
{
  "answer": "The main topic of the document is AI and machine learning.",
  "sources": ["document1.pdf", "document2.pdf"]
}
```
