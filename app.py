import subprocess
from flask import Flask, request, render_template, send_file, redirect, url_for
import os
from werkzeug.utils import secure_filename
from modules.setup import setup_nltk_resources, suppress_warnings
from modules.download_audio import download_audio_from_youtube
from modules.topic_modeling import build_lda_model, preprocess_and_split
from modules.transcribe_audio import transcribe_audio
from modules.summarize_text import summarize_text
from modules.ner import perform_ner
from modules.keyword_extraction import extract_keywords
import markdown
from markdown.extensions.toc import TocExtension
import json

app = Flask(__name__)
app.secret_key = 'YOUR_SECRET_KEY'

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
OUTPUT_DIR = "data/output"
MINDMAP_DIR = "data/mindmaps"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_DIR, MINDMAP_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_old_results():
    # Clear audio files
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    # Clear mindmap files
    for filename in os.listdir(MINDMAP_DIR):
        file_path = os.path.join(MINDMAP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
        
    # Clear uploaded files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def convert_video_to_audio(video_path):
    """Convert video to audio using FFmpeg directly"""
    audio_path = os.path.join(OUTPUT_DIR, "extracted_audio.mp3")
    try:
        subprocess.run([
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            audio_path
        ], check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return None
    except Exception as e:
        print(f"Error converting video: {e}")
        return None

def generate_mindmap_data(summary, entities, keywords, topics):
    """Generate data structure for mindmap visualization"""
    mindmap = {
        "name": "Video Analysis",
        "children": []
    }
    
    # Add summary as central node
    if summary:
        mindmap["children"].append({
            "name": "Summary",
            "children": [{"name": summary[:100] + "..."}]
        })
    
    # Add entities
    if entities:
        entity_list = [{"name": e} for e in entities.split('\n') if e.strip()]
        mindmap["children"].append({
            "name": "Named Entities",
            "children": entity_list
        })
    
    # Add keywords
    if keywords:
        keyword_list = [{"name": k} for k in keywords.split('\n') if k.strip()]
        mindmap["children"].append({
            "name": "Keywords",
            "children": keyword_list
        })
    
    # Add topics
    if topics:
        topic_list = [{"name": t} for t in topics.split('\n') if t.strip()]
        mindmap["children"].append({
            "name": "Topics",
            "children": topic_list
        })
    
    return mindmap

def process_audio_file(audio_path):
    """Common processing steps for both YouTube and uploaded files"""
    # Step 2: Transcribe
    transcription, detected_language = transcribe_audio(audio_path)

    # Step 3: Summarize
    summary = summarize_text(transcription)

    # Step 4: Named Entity Recognition (NER)
    entities = perform_ner(transcription)
    entity_str = "\n".join([f"Entity: {e[0]}, Label: {e[1]}" for e in entities])

    # Step 5: Keyword Extraction
    keywords = extract_keywords(transcription)
    keyword_str = "\n".join([f"{k[0]}: {k[1]}" for k in keywords])

    # Step 6: Topic Modeling
    processed_text = preprocess_and_split(transcription)
    lda_topics = build_lda_model(processed_text)
    topics_str = ""
    for i, topic in lda_topics.show_topics(num_topics=-1, num_words=3, formatted=False):
        topic_words = ", ".join([w for w, _ in topic])
        topics_str += f"Topic {i + 1}: {topic_words}\n"

    # Generate mindmap data
    mindmap_data = generate_mindmap_data(summary, entity_str, keyword_str, topics_str)
    mindmap_filename = os.path.join(MINDMAP_DIR, "mindmap.json")
    with open(mindmap_filename, 'w') as f:
        json.dump(mindmap_data, f)

    return {
        'transcription': transcription,
        'summary': summary,
        'entities': entity_str,
        'keywords': keyword_str,
        'topics': topics_str,
        'detected_language': detected_language,
        'audio_url': '/get_audio',
        'mindmap_url': '/mindmap'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_podcast():
    setup_nltk_resources()
    suppress_warnings()
    clear_old_results()

    try:
        audio_path = None
        
        # Check if YouTube URL was provided
        youtube_url = request.form.get('youtube_url')
        if youtube_url and youtube_url.strip():
            # Process YouTube URL
            download_audio_from_youtube(youtube_url)
            audio_path = os.path.join(OUTPUT_DIR, "video.mp3")
        else:
            # Check if file was uploaded
            if 'file_upload' not in request.files:
                return render_template('index.html', error="No file or YouTube URL provided")
            
            file = request.files['file_upload']
            if file.filename == '':
                return render_template('index.html', error="No selected file")
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Check if it's a video file
                if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    # Convert video to audio
                    audio_path = convert_video_to_audio(file_path)
                    if not audio_path:
                        return render_template('index.html', error="Failed to extract audio from video")
                else:
                    # It's an audio file
                    audio_path = file_path

        if not audio_path or not os.path.exists(audio_path):
            return render_template('index.html', error="Could not process the audio file")

        # Process the audio file
        results = process_audio_file(audio_path)
        
        return render_template('index.html',
                               audio_url=results['audio_url'],
                               transcription=results['transcription'],
                               summary=results['summary'],
                               entities=results['entities'],
                               keywords=results['keywords'],
                               topics=results['topics'],
                               detected_language=results['detected_language'],
                               mindmap_url=results['mindmap_url'],
                               processing_complete=True)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/get_audio')
def get_audio():
    # First check for YouTube processed audio
    audio_path = os.path.join(OUTPUT_DIR, "video.mp3")
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=False)
    
    # Then check for extracted audio from video
    extracted_audio = os.path.join(OUTPUT_DIR, "extracted_audio.mp3")
    if os.path.exists(extracted_audio):
        return send_file(extracted_audio, as_attachment=False)
    
    # Finally check for direct audio file uploads
    uploads = os.listdir(UPLOAD_FOLDER)
    for filename in uploads:
        if filename.lower().endswith(('.mp3', '.wav')):
            return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=False)
    
    return "Audio file not found", 404

@app.route('/mindmap')
def show_mindmap():
    mindmap_path = os.path.join(MINDMAP_DIR, "mindmap.json")
    if not os.path.exists(mindmap_path):
        return redirect(url_for('index'))
    
    with open(mindmap_path, 'r') as f:
        mindmap_data = json.load(f)
    
    return render_template('mindmap.html', mindmap_data=json.dumps(mindmap_data))

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)


    # good day 