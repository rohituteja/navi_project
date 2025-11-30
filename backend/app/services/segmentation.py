import os
import json
from openai import OpenAI
from typing import List, Dict, Any

def detect_segments(transcript: dict, telemetry: dict, offset_sec: float) -> List[Dict[str, Any]]:
    """
    Detects flight segments (Taxi, Takeoff, Climb, Cruise, Approach, Landing)
    using OpenAI's LLM to analyze the aligned transcript and telemetry summary.
    """
    print("Starting flight segmentation...")
    
    # 1. Prepare Telemetry Summary (sample every 30 seconds to reduce token count)
    telemetry_summary = []
    data_points = telemetry.get("data", [])
    
    if not data_points:
        print("No telemetry data found for segmentation")
        return []
        
    # Sample rate: every 30 seconds
    sample_interval = 30
    last_time = -sample_interval
    
    for point in data_points:
        if point["time_sec"] - last_time >= sample_interval:
            telemetry_summary.append(
                f"T={int(point['time_sec'])}s: Alt={int(point['alt_agl'])}ft, "
                f"Spd={int(point['ias'])}kt, RPM={int(point['rpm'])}, "
                f"Hdg={int(point['heading'])}°, VS={int(point['v_spd'])}fpm"
            )
            last_time = point["time_sec"]
            
    telemetry_text = "\n".join(telemetry_summary)
    
    # 2. Prepare Transcript
    transcript_text = ""
    if transcript and "segments" in transcript:
        for seg in transcript["segments"]:
            # Adjust time by offset to match telemetry time
            start_time = seg["start"] + offset_sec
            end_time = seg["end"] + offset_sec
            transcript_text += f"[{int(start_time)}s - {int(end_time)}s]: {seg['text']}\n"
            
    # 3. Construct Prompt
    prompt = f"""
    You are a flight instructor analyzing a flight recording.
    Your task is to segment the flight into distinct phases based on the provided telemetry summary and cockpit audio transcript.
    
    The available phases are:
    - PREFLIGHT (Engine start, checklists, runup)
    - TAXI (Movement on ground before takeoff)
    - TAKEOFF (Start of takeoff roll to initial climb)
    - CLIMB (Climbing to cruise altitude)
    - CRUISE (Level flight or transit)
    - MANEUVERS (Practice maneuvers like steep turns, stalls, etc.)
    - APPROACH (Descent and pattern entry)
    - LANDING (Final approach to touchdown)
    - TAXI_BACK (Taxi after landing)
    - SHUTDOWN (Engine shutdown)
    
    Here is the Telemetry Summary (sampled every 30s):
    {telemetry_text}
    
    Here is the Transcript (time-synced to telemetry):
    {transcript_text}
    
    Instructions:
    1. Analyze the data to identify the start and end times of each phase.
    2. Use the transcript to understand context (e.g., "Clear for takeoff", "Turning base").
    3. Use telemetry to confirm flight state (e.g., High RPM + Increasing Speed = Takeoff).
    4. Return a JSON array of segments. Each segment must have:
       - "name": Phase name (from the list above)
       - "start_time": Start time in seconds (integer)
       - "end_time": End time in seconds (integer)
       - "description": Brief description of what happened in this phase (1 sentence).
       
    The output must be VALID JSON only. Do not include markdown formatting.
    Example:
    [
        {{"name": "TAXI", "start_time": 0, "end_time": 120, "description": "Taxiing to runway 27."}},
        {{"name": "TAKEOFF", "start_time": 120, "end_time": 150, "description": "Takeoff roll and rotation."}}
    ]
    """
    
    # 4. Call OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping segmentation")
        return []
        
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful flight analysis assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Handle potential wrapper keys like {"segments": [...]}
        if isinstance(result, dict):
            if "segments" in result:
                segments = result["segments"]
            else:
                # Try to find a list value
                found_list = False
                for key, value in result.items():
                    if isinstance(value, list):
                        segments = value
                        found_list = True
                        break
                if not found_list:
                    print("Could not find segments list in JSON response")
                    return []
        elif isinstance(result, list):
            segments = result
        else:
            print("Unexpected JSON structure")
            return []
            
        print(f"Detected {len(segments)} segments")
        return segments
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return []

