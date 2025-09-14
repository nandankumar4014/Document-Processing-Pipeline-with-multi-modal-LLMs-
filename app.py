import os
import json
import io
import base64
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
from groq import Groq
import re

# Set page configuration
st.set_page_config(
    page_title="Insurance Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None

# --- Utility Functions ---
def preprocess_image(pil_image):
    """Preprocess image for better OCR results"""
    img_np = np.array(pil_image.convert('L'))
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def call_vision_model_for_description(base64_image, prompt="print the exact content of this image without description.", api_key=None):
    """Call Groq vision model for image description"""
    if not api_key:
        st.error("Groq API key not found.")
        return None

    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling vision model: {e}")
        return None

def get_line_level_boxes(pil_image):
    """Extract line-level bounding boxes using OCR"""
    img = np.array(pil_image)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    lines = {}
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50 and data['text'][i].strip():
            key = (data['page_num'][i], data['block_num'][i], data['par_num'][i], data['line_num'][i])
            if key not in lines:
                lines[key] = []
            lines[key].append(i)

    line_boxes = []
    for indices in lines.values():
        x = min([data['left'][i] for i in indices])
        y = min([data['top'][i] for i in indices])
        w = max([data['left'][i] + data['width'][i] for i in indices]) - x
        h = max([data['top'][i] + data['height'][i] for i in indices]) - y
        text = " ".join([data['text'][i] for i in indices])
        line_boxes.append({'text': text.strip(), 'box': (x, y, w, h)})
    
    return line_boxes

# --- LLM Utilities ---
def ask_llm(prompt, max_tokens=512, api_key=None):
    """Query the LLM with a prompt"""
    if not api_key:
        st.error("Groq API key not found.")
        return None

    client = Groq(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

# --- Document Segmentation ---
def detect_headers(full_text, api_key):
    """Detect headers and their hierarchical structure"""
    prompt = f"""
    Analyze this document text and identify all headers and their hierarchical structure.

    Document text:
    {full_text}

    Return a JSON structure representing the document's outline with headers and subheaders.
    Format your response as:

    [
      "Header 1",
      {{"Header 2": ["Subheader 1", "Subheader 2"]}},
      "Header 3"
    ]

    Only include the JSON array, no additional text.
    """

    response = ask_llm(prompt, max_tokens=1024, api_key=api_key)
    
    if not response:
        return ["Document"]

    try:
        # Clean and parse JSON response
        cleaned_response = response.strip()
        if '[' in cleaned_response:
            cleaned_response = cleaned_response[cleaned_response.find('['):]
        if ']' in cleaned_response:
            cleaned_response = cleaned_response[:cleaned_response.rfind(']')+1]

        try:
            headers = json.loads(cleaned_response)
            return headers
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_match = re.search(r'\[\s*.\s\]', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                try:
                    headers = json.loads(json_str)
                    return headers
                except:
                    return ["Document"]
            else:
                return ["Document"]
    except Exception as e:
        st.error(f"Error parsing headers: {e}")
        return ["Document"]

def segment_document(full_text, headers_structure):
    """Segment document based on detected headers"""
    segments = {}

    if not headers_structure:
        segments["Full Document"] = full_text.split('\n')
        return segments

    # Build list of all headers
    all_headers = []
    for item in headers_structure:
        if isinstance(item, str):
            all_headers.append(item)
        elif isinstance(item, dict):
            for main_header, subheaders in item.items():
                all_headers.append(main_header)
                if isinstance(subheaders, list):
                    all_headers.extend(subheaders)

    if not all_headers:
        segments["Full Document"] = full_text.split('\n')
        return segments

    all_headers.sort(key=len, reverse=True)
    current_header = "Introduction"
    segments[current_header] = []

    lines = full_text.split('\n')
    for line in lines:
        is_header = False
        for header in all_headers:
            if header and header.lower() in line.lower():
                current_header = header
                segments.setdefault(current_header, [])
                is_header = True
                break

        if not is_header and line.strip():
            segments.setdefault(current_header, []).append(line.strip())

    return segments

def extract_metadata(full_text, field, api_key):
    """Extract specific metadata field from document text"""
    if field == "Policy Number":
        prompt_template = f"""
        Extract ONLY the policy number from the following text. Look for "Policy Number:" specifically.
    
        Rules:
        1. Find the exact text after "Policy Number:"
        2. Return ONLY that specific number
        3. Ignore Master Policy Number, Certificate Number, Application Number
        4. Return just the digits/number, nothing else
    
        Examples:
        Text: Master Policy Number: 2700001636 Certificate Number: 7 Policy Number: 2700001636
        Answer: 2700001636

        Text: {full_text}
        """
    elif field == "Mobile Number":
        prompt_template = f"""
        Extract the mobile number from the following text. Return only the 10-digit mobile number. If the mobile number is not found, return 'null'.

        Text: {full_text}
        """
    else:
        prompt_template = f"""
        Extract the {field} from the following text. Return only the value. If the value is not found, return 'null'.

        Text: {full_text}
        """

    response = ask_llm(prompt_template, max_tokens=32, api_key=api_key)
    
    if not response:
        return None
        
    response = response.strip().lower()
    if response == 'null' or not response:
        return None
    else:
        return response

def highlight_best_match(pil_image, line_boxes, answer_text):
    """Highlight best matching text in image"""
    img = np.array(pil_image.convert("RGB"))
    matches_above_threshold = []

    has_table_structure = '|' in answer_text or 'table' in answer_text.lower()
    match_threshold = 90 if has_table_structure else 80
    answer_lines = answer_text.split('\n')

    for item in line_boxes:
        best_score = 0
        if has_table_structure:
            for line in answer_lines:
                if line.strip():
                    score = fuzz.token_set_ratio(line.lower(), item['text'].lower())
                    best_score = max(best_score, score)
        else:
            score = fuzz.token_set_ratio(answer_text.lower(), item['text'].lower())
            best_score = score

        if best_score > match_threshold:
            matches_above_threshold.append(item)

    if matches_above_threshold:
        for item in matches_above_threshold:
            x, y, w, h = item['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return Image.fromarray(img), True
    else:
        return Image.fromarray(img), False

def process_pdf(uploaded_file, api_key):
    """Process uploaded PDF file"""
    # Save uploaded file temporarily
    with open("temp_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Convert PDF to images
        pil_images = convert_from_path("temp_pdf.pdf", dpi=300)
        if not pil_images:
            st.error("No pages found in PDF.")
            return None

        pages_text = []
        preprocessed_images = []
        line_boxes_per_page = []

        # Process all pages
        progress_bar = st.progress(0)
        for i, page in enumerate(pil_images):
            progress_bar.progress((i + 1) / len(pil_images))
            
            pre_img = preprocess_image(page)
            preprocessed_images.append(pre_img)

            text = pytesseract.image_to_string(np.array(pre_img))
            pages_text.append(text)

            line_boxes = get_line_level_boxes(pre_img)
            line_boxes_per_page.append(line_boxes)

        # Process document
        full_text = "\n".join(pages_text)
        
        # Extract metadata
        metadata = {}
        metadata_fields = ["policy number", "policy start date", "policy end date", "policy holder name"]
        for field in metadata_fields:
            metadata[field] = extract_metadata(full_text, field, api_key)

        # Detect headers and segment document
        headers_structure = detect_headers(full_text, api_key)
        document_segments = segment_document(full_text, headers_structure)

        return {
            'pages_text': pages_text,
            'preprocessed_images': preprocessed_images,
            'line_boxes_per_page': line_boxes_per_page,
            'full_text': full_text,
            'metadata': metadata,
            'headers_structure': headers_structure,
            'document_segments': document_segments
        }
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists("temp_pdf.pdf"):
            os.remove("temp_pdf.pdf")

def main():
    st.title("DOCUHAWK")
    st.title("📄 Insurance Document Analyzer")
    st.markdown("Upload an insurance PDF document to analyze and extract information.")

    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙ Configuration")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        
        if not api_key:
            st.warning("Please enter your Groq API key to proceed.")
            st.stop()

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("🔄 Process PDF"):
            with st.spinner("Processing PDF... This may take a few minutes."):
                processed_data = process_pdf(uploaded_file, api_key)
                if processed_data:
                    st.session_state.processed_data = processed_data
                    st.success("PDF processed successfully!")

    # Display results if data is available
    if st.session_state.processed_data:
        data = st.session_state.processed_data
        
        # Navigation buttons
        st.subheader("📋 Document Analysis Options")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            preview_btn = st.button("🖼 Preview Pages", use_container_width=True)
        
        with col2:
            segments_btn = st.button("📑 Segments", use_container_width=True)
        
        with col3:
            metadata_btn = st.button("📊 Metadata", use_container_width=True)
        
        with col4:
            headers_btn = st.button("📋 Headers & Subheaders", use_container_width=True)
        
        with col5:
            qa_btn = st.button("❓ Q&A Analysis", use_container_width=True)

        # Display content based on button clicks
        if preview_btn:
            st.subheader("🖼 PDF Pages Preview")
            st.write(f"*Total Pages*: {len(data['preprocessed_images'])}")
            
            # Display all pages
            for i, img in enumerate(data['preprocessed_images']):
                st.write(f"*Page {i + 1}*")
                st.image(img, caption=f"Page {i + 1}", use_container_width=True)
                st.divider()

        elif segments_btn:
            st.subheader("📑 Document Segments")
            st.write(f"*Total Segments Found*: {len(data['document_segments'])}")
            
            for segment_name, content in data['document_segments'].items():
                with st.expander(f"📄 {segment_name}", expanded=False):
                    if content:
                        segment_text = "\n".join(content)
                        st.text_area(
                            f"Content for {segment_name}",
                            value=segment_text,
                            height=200,
                            key=f"segment_{segment_name}"
                        )
                        st.info(f"*Lines in this segment*: {len(content)}")
                    else:
                        st.warning("This segment is empty.")

        elif metadata_btn:
            st.subheader("📊 Extracted Metadata")
            
            # Create a nice display for metadata
            for field, value in data['metadata'].items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"{field.title()}:")
                with col2:
                    if value:
                        st.success(value)
                    else:
                        st.error("Not found")
            
            # Show raw metadata in an expandable section
            with st.expander("🔍 Raw Metadata JSON", expanded=False):
                st.json(data['metadata'])

        elif headers_btn:
            st.subheader("📋 Headers & Subheaders Structure")
            
            # Display headers structure
            st.write("*Document Outline:*")
            
            def display_headers_recursive(headers, level=0):
                """Recursively display headers with proper indentation"""
                indent = "  " * level
                
                for item in headers:
                    if isinstance(item, str):
                        if level == 0:
                            st.markdown(f"### {indent}📌 {item}")
                        elif level == 1:
                            st.markdown(f"#### {indent}📍 {item}")
                        else:
                            st.markdown(f"{indent}• {item}")
                    elif isinstance(item, dict):
                        for main_header, subheaders in item.items():
                            if level == 0:
                                st.markdown(f"### {indent}📌 {main_header}")
                            else:
                                st.markdown(f"#### {indent}📍 {main_header}")
                            
                            if isinstance(subheaders, list):
                                for subheader in subheaders:
                                    st.markdown(f"{indent}  • {subheader}")
            
            if data['headers_structure']:
                display_headers_recursive(data['headers_structure'])
            else:
                st.warning("No clear header structure detected in the document.")
            
            # Show raw structure in expandable section
            with st.expander("🔍 Raw Headers JSON", expanded=False):
                st.json(data['headers_structure'])

        elif qa_btn or 'show_qa' in st.session_state:
            # Set flag to show Q&A section
            st.session_state.show_qa = True
            
        # Show Q&A section if button clicked or flag is set
        if 'show_qa' in st.session_state and st.session_state.show_qa:
            st.subheader("❓ Ask Questions")
        
            # Predefined questions
            predefined_questions = {
                "Policy Number": "Extract the policy number",
                "Policy Holder Name": "Extract the policy holder name", 
                "Policy Commencement Date": "Extract the policy commencement date",
                "Policy End Date": "Extract the policy end date",
                "Nominee Name": "Extract the nominee name",
                "Policy Holder Details": "Extract policy holder details",
                "Nominee Details": "Extract nominee details", 
                "Address": "Extract the address",
                "Mobile Number": "Extract the mobile number",
                "Type of Policy": "Extract the type of policy"
            }

            # Question selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                question_type = st.selectbox("Select a question type:", 
                                           ["Custom Question"] + list(predefined_questions.keys()))
            
            if question_type == "Custom Question":
                question = st.text_input("Enter your custom question:")
            else:
                question = predefined_questions[question_type]
                st.write(f"*Question*: {question}")

            if st.button("🔍 Get Answer") and question:
                with st.spinner("Analyzing document..."):
                    # Check if this is a predefined question that needs special handling
                    if question_type in ["Policy Number", "Mobile Number"]:
                        # Use the same specific extraction logic as metadata
                        answer = extract_metadata(data['full_text'], question_type, api_key)
                        if answer:
                            answer = str(answer)  # Convert to string for consistency
                        else:
                            answer = "Not found in the document"
                    else:
                        # Create prompt for LLM for other questions
                        if question_type == "Policy Holder Name":
                            prompt = f"""
Extract ONLY the policy holder name from the following text. Look for "Policy Holder Name:" or "Insured Name:" specifically.

Rules:
1. Find the exact text after "Policy Holder Name:" or "Insured Name:"
2. Return ONLY that specific name
3. Return just the name, nothing else

Text: {data['full_text']}
"""
                        elif question_type == "Policy Commencement Date" or question_type == "Policy Start Date":
                            prompt = f"""
Extract ONLY the policy commencement date or policy start date from the following text.

Rules:
1. Look for "Policy Commencement Date:", "Policy Start Date:", "Effective Date:", or similar
2. Return ONLY the date
3. Return just the date, nothing else

Text: {data['full_text']}
"""
                        elif question_type == "Policy End Date":
                            prompt = f"""
Extract ONLY the policy end date or policy expiry date from the following text.

Rules:
1. Look for "Policy End Date:", "Policy Expiry Date:", "Maturity Date:", or similar
2. Return ONLY the date
3. Return just the date, nothing else

Text: {data['full_text']}
"""
                        elif question_type == "Nominee Name":
                            prompt = f"""
Extract ONLY the nominee name from the following text.

Rules:
1. Look for "Nominee Name:", "Nominee:", or similar
2. Return ONLY the nominee name
3. Return just the name, nothing else

Text: {data['full_text']}
"""
                        elif question_type == "Address":
                            prompt = f"""
Extract the complete address from the following text.

Rules:
1. Look for address information (usually after "Address:", "Permanent Address:", etc.)
2. Return the complete address
3. Include all address lines if multiple lines exist

Text: {data['full_text']}
"""
                        elif question_type == "Type of Policy":
                            prompt = f"""
Extract the type of policy from the following text.

Rules:
1. Look for "Policy Type:", "Type of Policy:", "Plan:", or similar
2. Return ONLY the policy type
3. Return just the type, nothing else

Text: {data['full_text']}
"""
                        else:
                            # Generic prompt for custom questions
                            prompt = f"""
You are given OCR text extracted from an insurance document.

OCR Text:
{data['full_text']}

Now answer this question: "{question}"

If the answer is a table or structured data, return the COMPLETE table structure with all rows and columns.
Return only the *exact content* from the OCR text that answers the question. Do not paraphrase or explain.
"""

                        answer = ask_llm(prompt, max_tokens=512, api_key=api_key)
                    
                    if answer:
                        st.subheader("📝 Answer")
                        st.write(answer)

                        # Find best matching page and highlight
                        best_page_idx = -1
                        best_score = 0
                        
                        for i, line_boxes in enumerate(data['line_boxes_per_page']):
                            score = 0
                            for item in line_boxes:
                                score += fuzz.token_set_ratio(answer.lower(), item['text'].lower())
                            if score > best_score:
                                best_score = score
                                best_page_idx = i

                        if best_page_idx != -1:
                            highlighted_img, matched = highlight_best_match(
                                data['preprocessed_images'][best_page_idx], 
                                data['line_boxes_per_page'][best_page_idx], 
                                answer
                            )

                            if matched:
                                st.subheader(f"📍 Highlighted Answer (Page {best_page_idx + 1})")
                                st.image(highlighted_img, caption=f"Page {best_page_idx + 1}: Highlighted Answer", use_container_width=True)
                                
                                # Option to check other pages
                                if st.checkbox("Check other pages for this answer"):
                                    for i in range(len(data['preprocessed_images'])):
                                        if i != best_page_idx:
                                            highlighted_img, matched = highlight_best_match(
                                                data['preprocessed_images'][i], 
                                                data['line_boxes_per_page'][i], 
                                                answer
                                            )
                                            if matched:
                                                st.image(highlighted_img, caption=f"Page {i + 1}: Additional Match", use_column_width=True)
                            else:
                                st.warning("No strong match found in the document images.")
                    else:
                        st.error("Could not extract the requested information from the document.")

if __name__ == "__main__":
    main()