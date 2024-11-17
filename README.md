# PDF📖 - Document Query Chat Bot🤖

*The **PDF - Document Query Chat Bot** allows users to upload PDF files and interact with the content through a chatbot interface. It leverages natural language processing (NLP) to enable users to ask questions, summarize content, or retrieve specific information from the uploaded PDFs. This app is ideal for anyone who needs to quickly extract insights or explore large PDF documents interactively.*<br>

![alt text](assets/image.png)

## Features<br>
* **PDF Upload:** Users can upload a PDF file to the app.<br>
* **Interactive Chat:** Once the PDF is uploaded, users can chat with the app to extract relevant information, get summaries, or ask specific questions.<br>
* **Text Extraction:** The app reads the content of the PDF and makes it searchable.<br>
* **Contextual Responses:** The chatbot provides responses based on the content of the uploaded PDF.<br>

## Technologies Used<br>
* **Frontend:** Gradio<br>
* **Backend:** Python<br>
* **NLP:** Phi-3-mini-4k-instruct-q4.gguf or similar language models for text-based interaction.<br>
* **Real-time Processing:** Instantaneous interaction with the document content.<br>
* **PDF Processing:** PyPDFLoader and other Python libraries for text extraxtion.<br>
* **File Storage:** Chroma storage for PDFs.<br>

## Getting Started
### Prerequisites
* Python 3.11+<br>
* Gradio (for frontend)<br>
* Dependencies (listed below)<br>

### Installation
1. Clone the repositery:<br>
git clone https://github.com/vishwajeetsingh01/document_query_project.git<br>
cd document_query_project

2. Create a virtual environment in the terminal:<br>
python -m venv venv

3. Activate virtual environment:<br>
.\venv\Scripts\activate

**Note:** *If you get error than fire the below commond: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass*

4. Install dependencies:<br>
pip install -r requirements.txt

5. Run the application:<br>
python main.py

Now, the app should be running on locally at http://127.0.0.1:7860/.

## File Upload
To interact with the PDF, simply click on the Upload PDF button on the frontend, select a file, and the chat interface will be ready to respond based on the content of the document.

## Usage
**Chat with PDF:** After uploading a PDF, ask questions related to its contents (e.g., "What is the summary of this document?", "Tell me about the second chapter.").<br>
**Extract Information:** You can ask for specific data or quotes directly (e.g., "What is the value of X in section 3?").<br>
**Summarization:** Ask the bot to provide a concise summary of the entire document or specific sections.<br>

## Contributing
Contributions are welcome! If you find any issues or would like to add new features, please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a pull request

## Achnowledgements
* Special thanks to GPT4All for providing powerful language models.<br>
* Thanks to the maintainers of PyPDFLoader for their contributions to PDF processing.<br>
* Inspired by modern AI-driven tools for document analysis and interaction.

  
