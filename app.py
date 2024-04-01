
from flask import Flask, request, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
import os
app = Flask(__name__)
# Loading BART model
tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#Facebook dataset
model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
def summarize_conversation_bart(meeting):
    # Concatenating meeting convo into a single string
    meeting_text = " ".join(meeting)
    
    # Tokenizing the text 
    inputs = tokenizer_bart([meeting_text], max_length=10000, return_tensors='pt', truncation=True)
    
    # Generating summary
    summary_ids = model_bart.generate(inputs['input_ids'], max_length=1000, min_length=20, length_penalty=3.0, num_beams=5, early_stopping=False)
    
    # Decoding the summary
    summary_text = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text
from docx import Document
def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    text = ""
    if file_extension == '.docx':
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    return text
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/summarize', methods=['POST'])
def summarize():
    # Check if meeting text is provided
    meeting_text = request.form.get('meeting_text')
    # If meeting text is not provided, check if a file is uploaded
    if not meeting_text:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                # Save the uploaded file temporarily
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                
                # Extract text from the file
                meeting_text = extract_text_from_file(file_path)
                
                # Remove the temporary file
                os.remove(file_path)
    # If neither meeting text nor file is provided, render error message
    if not meeting_text:
        return render_template('index.html', error='Please enter meeting data or upload a file.')
    # Process and summarize the meeting text
    meetings = [line.strip() for line in meeting_text.split('\n') if line.strip()]
    summary = summarize_conversation_bart(meetings)
    return render_template('index.html', meeting_text=meeting_text, summary=summary)
if __name__ == '__main__':
    app.run(debug=True)
