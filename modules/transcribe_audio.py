import whisper
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)  # Ensure sentence tokenizer is ready

def transcribe_audio(audio_path):
    """Transcribe audio and return (transcription, language)
    
    Args:
        audio_path (str): Path to audio file to transcribe
        
    Returns:
        tuple: (transcription_text, detected_language) or (error_message, "error")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and transcribe
        model = whisper.load_model("base").to(device)
        result = model.transcribe(audio_path)
        
        detected_language = result.get("language", "unknown")
        transcription = result.get("text", "")
        
        if not transcription:
            return "No speech detected in audio", "empty"

        # Translate non-English content if needed
        if detected_language != "en":
            try:
                translator = pipeline(
                    "translation", 
                    model="Helsinki-NLP/opus-mt-mul-en",
                    device=0 if device == "cuda" else -1
                )
                
                sentences = sent_tokenize(transcription)
                translated_chunks = []
                
                for sentence in sentences:
                    try:
                        translated = translator(sentence, max_length=512)[0]['translation_text']
                        translated_chunks.append(translated)
                    except Exception as e:
                        translated_chunks.append(sentence)  # Fallback to original
                        
                transcription = ' '.join(translated_chunks)
                detected_language = "en"  # Mark as translated to English
                
            except Exception as e:
                # Continue with original text if translation fails
                pass
            
        return transcription, detected_language
        
    except Exception as e:
        return f"Transcription failed: {str(e)}", "error"
