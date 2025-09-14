📄 DocuHawk: Insurance Document Analyzer
DocuHawk is a Streamlit application designed to streamline the process of analyzing insurance PDF documents. It combines the power of Optical Character Recognition (OCR) with Large Language Models (LLMs) to extract, segment, and answer questions about document content, providing a quick and efficient way to find key information.

✨ Features
PDF Processing: Upload a PDF insurance document and automatically process it to extract text and images.

OCR Integration: Utilizes pytesseract to convert document pages into machine-readable text.

Advanced Text Extraction: Leverages the Groq vision model to accurately capture text from document pages, including structured data like tables.

Smart Metadata Extraction: Automatically identifies and extracts critical metadata such as policy number, policy holder name, and policy dates.

Document Segmentation: Uses an LLM to detect headers and subheaders, segmenting the document into logical sections for easier navigation.

Interactive Q&A: Ask natural language questions about the document's content and get concise, accurate answers.

Visual Highlighting: The application finds and highlights the exact location of the answer in the original document image, providing visual verification.

User-friendly Interface: Built with Streamlit for a clean, intuitive web interface.

🚀 How to Run Locally
Prerequisites
You'll need a Groq API key to use the LLM features. You can get one for free from the Groq Console.

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/docuhawk.git
cd docuhawk
Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:

Bash

pip install -r requirements.txt
(Note: You will need to create a requirements.txt file with the libraries used in the code. A suggested list is provided below).

Install Tesseract OCR:
This application depends on Tesseract OCR. Follow the installation instructions for your operating system:

Windows: Download the installer from the Tesseract GitHub page. Make sure to add the installation directory to your system's PATH.

macOS: Use Homebrew: brew install tesseract

Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr

Running the App
Set your Groq API key as an environment variable or enter it directly in the app's sidebar.

Bash

export GROQ_API_KEY="your_groq_api_key_here"
Run the Streamlit application:

Bash

streamlit run app.py
(Assuming the Python script is named app.py).

The app will open in your default web browser.

🛠️ Requirements.txt
To ensure others can easily install the necessary dependencies, create a requirements.txt file with the following content:

Plaintext

streamlit
pytesseract
Pillow
opencv-python
pdf2image
fuzzywuzzy
python-Levenshtein
groq
⚙️ Configuration
Groq API Key: This is required for the LLM-powered features. Enter it in the sidebar.

Tesseract Path: If Tesseract is not in your system's PATH, you may need to specify its location in your script using pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'.

🤝 Contributing
Contributions are welcome! If you find a bug or have a suggestion for a new feature, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
