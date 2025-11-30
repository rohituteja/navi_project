import shutil
import uuid
import subprocess
import json
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job store for MVP
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    transcript: Optional[dict] = None

@router.post("/upload")
async def upload_files(
    telemetry_file: UploadFile = File(...),
    audio_file: Optional[UploadFile] = File(None)
):
    job_id = str(uuid.uuid4())
    
    # Validate telemetry extension
    if not telemetry_file.filename.endswith(('.xlsx', '.csv')):
        raise HTTPException(status_code=400, detail="Invalid telemetry file format. Must be .xlsx or .csv")
    
    # Save telemetry
    telemetry_path = os.path.join(UPLOAD_DIR, f"{job_id}_telemetry_{telemetry_file.filename}")
    with open(telemetry_path, "wb") as buffer:
        shutil.copyfileobj(telemetry_file.file, buffer)
        
    audio_path = None
    if audio_file:
        # Validate audio extension
        if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
             raise HTTPException(status_code=400, detail="Invalid audio file format. Must be .mp3, .wav, or .m4a")
        
        audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_audio_{audio_file.filename}")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    
    jobs[job_id] = {
        "status": "queued",
        "telemetry_path": telemetry_path,
        "audio_path": audio_path
    }
    
    return {"job_id": job_id, "status": "queued"}
    

def normalize_audio(input_path: str, output_path: str):
    """
    Normalize audio to 16kHz mono MP3 using ffmpeg.
    Using MP3 to stay under OpenAI's 25MB file size limit (WAV files are too large).
    """
    command = [
        "ffmpeg",
        "-y", # Overwrite output file
        "-i", input_path,
        "-ac", "1", # Mono
        "-ar", "16000", # 16kHz
        "-b:a", "64k", # 64kbps bitrate for MP3 (good quality for speech)
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise HTTPException(status_code=500, detail="Audio normalization failed")

@router.post("/transcribe")
async def transcribe_audio(job_id: str = Form(...)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    job["status"] = "transcribing"
    
    try:
        transcript = None
        # Store cache in project root directory for persistence
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        cached_transcript_path = os.path.join(project_root, "cached_transcript.json")
        
        if job["audio_path"]:
            # Check if cached transcript exists
            if os.path.exists(cached_transcript_path):
                print(f"Using cached transcript for job {job_id}")
                with open(cached_transcript_path, "r") as f:
                    transcript = json.load(f)
                job["used_cache"] = True
                job["message"] = "Used cached transcript"
            else:
                # Check if original file is small enough to use directly
                original_size = os.path.getsize(job["audio_path"])
                max_size = 25 * 1024 * 1024  # 25MB in bytes
                
                print(f"Original audio file size: {original_size / (1024*1024):.2f} MB")
                
                if original_size <= max_size:
                    # Use original file directly - no need to normalize
                    print(f"File is under 25MB, using original audio")
                    audio_path_to_send = job["audio_path"]
                else:
                    # File too large, normalize to compressed MP3
                    print(f"File exceeds 25MB, normalizing to compressed MP3")
                    normalized_path = os.path.join(UPLOAD_DIR, f"{job_id}_normalized.mp3")
                    normalize_audio(job["audio_path"], normalized_path)
                    audio_path_to_send = normalized_path
                
                # Call OpenAI Whisper API using official SDK
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
                
                client = OpenAI(api_key=api_key)
                
                print(f"Calling OpenAI Whisper API for job {job_id}...")
                print(f"Sending file: {audio_path_to_send}, size: {os.path.getsize(audio_path_to_send) / (1024*1024):.2f} MB")
                try:
                    # Use Path object - the SDK handles opening/reading the file
                    from pathlib import Path
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=Path(audio_path_to_send),
                        response_format="verbose_json"
                    )
                    
                    # Convert response to dict
                    transcript = response.model_dump()
                except Exception as api_error:
                    print(f"OpenAI API detailed error: {type(api_error).__name__}: {str(api_error)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Save transcript to cache
                with open(cached_transcript_path, "w") as f:
                    json.dump(transcript, f, indent=2)
                print(f"Saved transcript to cache: {cached_transcript_path}")
                
                # Clean up normalized file if we created one
                if original_size > max_size:
                    normalized_path = os.path.join(UPLOAD_DIR, f"{job_id}_normalized.mp3")
                    if os.path.exists(normalized_path):
                        os.remove(normalized_path)
                
                job["used_cache"] = False
                job["message"] = "Generated new transcript and cached it"
            
        else:
            # Load default transcript
            # For MVP, just return a hardcoded one if file not found
            # In real app, we would load from a file in the repo
            transcript = {
                "text": "Default transcript (no audio uploaded).", 
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "This is a default transcript segment."}
                ]
            }
            job["used_cache"] = False
            job["message"] = "No audio uploaded, using default transcript"
            
        job["transcript"] = transcript
        job["status"] = "done"
        return transcript
        
    except Exception as e:
        job["status"] = "error"
        job["message"] = str(e)
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/cache/status")
async def get_cache_status():
    """Check if cached transcript exists"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cached_transcript_path = os.path.join(project_root, "cached_transcript.json")
    
    if os.path.exists(cached_transcript_path):
        # Get file modification time
        mod_time = os.path.getmtime(cached_transcript_path)
        return {
            "cached": True,
            "path": cached_transcript_path,
            "modified_timestamp": mod_time
        }
    else:
        return {"cached": False}
