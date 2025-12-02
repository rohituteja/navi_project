import shutil
import uuid

import json
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from app.services.debrief import generate_debrief, generate_segment_debrief

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
    audio_file: Optional[UploadFile] = File(None),
    plane_type: str = Form("Sling Next Generation Trainer (NGT)")
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
        "audio_path": audio_path,
        "plane_type": plane_type
    }
    
    return {"job_id": job_id, "status": "queued"}
    



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
                    # Use original file directly
                    print(f"File is under 25MB, using original audio")
                    audio_path_to_send = job["audio_path"]
                else:
                    # File too large, but we removed normalization support
                    # For this specific project scope, we assume files are small enough or pre-processed
                    print(f"File exceeds 25MB. Normalization is disabled in this simplified version.")
                    # We'll try to send it anyway, but it might fail if OpenAI rejects it.
                    # Alternatively, we could raise an error here.
                    # Given the user requirement "the only audio file being used here will be a 20mb mp3 file",
                    # we can assume this path won't be hit or we can just proceed.
                    audio_path_to_send = job["audio_path"]
                
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



@router.post("/analyze")
async def analyze_flight(job_id: str = Form(...)):
    """
    Performs full flight analysis:
    1. Parse telemetry
    2. Get transcript (from job)
    3. Calculate alignment offset
    4. Detect flight segments
    5. Return comprehensive analysis
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job must be transcribed first")
    
    try:
        from app.services.telemetry import parse_telemetry
        from app.services.alignment import calculate_offset
        from app.services.segmentation import detect_segments
        
        # 1. Parse Telemetry
        print(f"Parsing telemetry for job {job_id}...")
        telemetry = parse_telemetry(job["telemetry_path"])
        
        # 2. Get Transcript
        transcript = job.get("transcript")
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        # 3. Calculate Alignment
        print(f"Calculating alignment offset...")
        alignment = calculate_offset(transcript, telemetry)
        
        # 4. Detect Segments
        print(f"Detecting flight segments...")
        plane_type = job.get("plane_type", "Unknown")
        segments = detect_segments(transcript, telemetry, alignment["offset_sec"], plane_type)
        
        # 5. Build Analysis Result
        analysis = {
            "job_id": job_id,
            "telemetry": {
                "metadata": telemetry["metadata"],
                "data_points": len(telemetry["data"])
            },
            "transcript": {
                "segment_count": len(transcript.get("segments", [])),
                "duration": transcript.get("duration", 0)
            },
            "alignment": alignment,
            "segments": segments,
            "status": "complete"
        }
        
        # Store in job
        job["analysis"] = analysis
        
        return analysis
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{job_id}")
async def get_analysis(job_id: str):
    """Get the analysis results for a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if "analysis" not in job:
        raise HTTPException(status_code=404, detail="Analysis not yet performed. Call /analyze first.")
    
    return job["analysis"]

@router.get("/telemetry/{job_id}")
async def get_telemetry_data(job_id: str, segment: str = None):
    """
    Get telemetry data for a job, optionally filtered by segment.
    If segment is provided, returns only data points within that segment's time range.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    try:
        from app.services.telemetry import parse_telemetry
        
        telemetry = parse_telemetry(job["telemetry_path"])
        
        if segment and "analysis" in job:
            # Filter by segment
            segments = job["analysis"].get("segments", [])
            target_segment = next((s for s in segments if s["name"] == segment), None)
            
            if target_segment:
                start_time = target_segment["start_time"]
                end_time = target_segment["end_time"]
                
                filtered_data = [
                    d for d in telemetry["data"]
                    if start_time <= d["time_sec"] <= end_time
                ]
                
                return {
                    "metadata": telemetry["metadata"],
                    "segment": target_segment,
                    "data": filtered_data
                }
        
        # Return all data
        return telemetry
        
    except Exception as e:
        print(f"Telemetry retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debrief")
async def get_flight_debrief(job_id: str = Form(...)):
    """
    Generates a short CFI-style debrief for the flight.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Check if analysis is done
    if "analysis" not in job:
         raise HTTPException(status_code=400, detail="Analysis must be completed first")
         
    # Check if debrief already exists
    if "debrief" in job:
        return {"debrief": job["debrief"]}
        
    try:
        analysis = job["analysis"]
        transcript = job.get("transcript")
        
        # Load telemetry
        from app.services.telemetry import parse_telemetry
        telemetry = parse_telemetry(job["telemetry_path"])
        
        debrief = generate_debrief(
            transcript=transcript,
            telemetry=telemetry,
            segments=analysis["segments"],
            plane_type=job.get("plane_type", "Unknown")
        )
        
        job["debrief"] = debrief
        return {"debrief": debrief}
        
    except Exception as e:
        print(f"Debrief error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debrief/segment")
async def get_segment_debrief(job_id: str = Form(...), segment_index: int = Form(...)):
    """
    Generates a focused CFI-style debrief for a specific segment.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Check if analysis is done
    if "analysis" not in job:
        raise HTTPException(status_code=400, detail="Analysis must be completed first")
    
    analysis = job["analysis"]
    segments = analysis.get("segments", [])
    
    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(status_code=400, detail="Invalid segment index")
    
    # Get target, previous, and next segments
    target_segment = segments[segment_index]
    prev_segment = segments[segment_index - 1] if segment_index > 0 else None
    next_segment = segments[segment_index + 1] if segment_index < len(segments) - 1 else None
    
    try:
        transcript = job.get("transcript")
        
        # Load telemetry
        from app.services.telemetry import parse_telemetry
        telemetry = parse_telemetry(job["telemetry_path"])
        
        debrief = generate_segment_debrief(
            transcript=transcript,
            telemetry=telemetry,
            target_segment=target_segment,
            prev_segment=prev_segment,
            next_segment=next_segment,
            plane_type=job.get("plane_type", "Unknown")
        )
        
        return {"debrief": debrief}
        
    except Exception as e:
        print(f"Segment debrief error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


