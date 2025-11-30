import os
import json
from openai import OpenAI



def preprocess_transcript(transcript: dict) -> str:
    """
    Extracts segments into a concise text format: [Start-End] Text
    """
    if not transcript or "segments" not in transcript:
        return ""
    
    lines = []
    for segment in transcript["segments"]:
        start = f"{segment['start']:.1f}"
        end = f"{segment['end']:.1f}"
        text = segment["text"].strip()
        lines.append(f"[{start}-{end}] {text}")
    
    return "\n".join(lines)

def preprocess_telemetry(telemetry: dict) -> str:
    """
    Downsamples telemetry and selects key columns to save tokens.
    Returns a CSV-like string.
    """
    if not telemetry or "data" not in telemetry:
        return ""
    
    data = telemetry["data"]
    if not data:
        return ""
    
    # Header
    lines = ["time,alt,speed,rpm,v_spd"]
    
    # Downsample to approx 0.5Hz (every 2 seconds) to save tokens, 
    # but keep enough resolution for events. 
    # Assuming original data is roughly 1Hz or higher.
    step = 2 
    
    for i in range(0, len(data), step):
        point = data[i]
        t = f"{point.get('time_sec', 0):.1f}"
        alt = f"{point.get('alt_agl', 0):.0f}"
        speed = f"{point.get('ias', 0):.0f}"
        rpm = f"{point.get('rpm', 0):.0f}"
        v_spd = f"{point.get('v_spd', 0):.0f}"
        
        lines.append(f"{t},{alt},{speed},{rpm},{v_spd}")
        
    return "\n".join(lines)

def detect_key_telemetry_events(telemetry: dict) -> str:
    """
    Analyzes telemetry to find key events like Run-ups and Takeoffs.
    Returns a formatted string description.
    """
    if not telemetry or "data" not in telemetry:
        return "No telemetry data available."
    
    data = telemetry["data"]
    events = []
    
    # 1. Detect Run-up (High RPM, Low Speed)
    # Look for RPM > 3000 while Speed < 30
    run_up_candidates = []
    for point in data:
        if point.get("rpm", 0) > 3000 and point.get("ias", 0) < 30:
            run_up_candidates.append(point["time_sec"])
            
    if run_up_candidates:
        # Group consecutive seconds
        # Simple clustering: if points are close, group them
        start = run_up_candidates[0]
        end = run_up_candidates[0]
        distinct_runups = []
        
        for t in run_up_candidates[1:]:
            if t - end < 5: # continuous event
                end = t
            else:
                if end - start > 2: # Minimum duration 2s
                    distinct_runups.append((start, end))
                start = t
                end = t
        if end - start > 2:
            distinct_runups.append((start, end))
            
        for start, end in distinct_runups:
            events.append(f"- Potential Engine Run-up (High RPM > 3000, Low Speed): {start:.1f}s to {end:.1f}s")

    # 2. Detect Takeoff Roll (Speed increasing past 40kt)
    takeoff_time = None
    for i in range(10, len(data)):
        point = data[i]
        prev = data[i-10]
        if point.get("ias", 0) > 40 and prev.get("ias", 0) < 30:
            takeoff_time = point["time_sec"]
            events.append(f"- Potential Takeoff Roll (Speed > 40kt): ~{takeoff_time:.1f}s")
            break
            
    if not events:
        return "No significant telemetry events detected automatically."
        
    return "\n".join(events)

def calculate_offset_with_llm(transcript: dict, telemetry: dict) -> dict:
    """
    Uses OpenAI's gpt-5-nano to determine the time offset.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Falling back to default offset.")
        return {"offset_sec": 0.0, "confidence": 0.0, "method": "fallback_no_key"}

    client = OpenAI(api_key=api_key)
    
    transcript_text = preprocess_transcript(transcript)
    telemetry_text = preprocess_telemetry(telemetry)
    key_events_text = detect_key_telemetry_events(telemetry)
    
    # Truncate if too long (rough safety check)
    if len(transcript_text) > 50000:
        transcript_text = transcript_text[:50000] + "...(truncated)"
    if len(telemetry_text) > 50000:
        telemetry_text = telemetry_text[:50000] + "...(truncated)"

    system_prompt = """
    You are an expert flight data analyst. Your task is to synchronize an audio transcript with flight telemetry data.
    
    The audio transcript has timestamps relative to the start of the recording.
    The telemetry data has timestamps relative to the start of the data recording.
    
    There is a time offset between them: Telemetry Time = Audio Time + Offset.
    
    **CRITICAL INSTRUCTION: GLOBAL CONSISTENCY CHECK**
    Do NOT just match the first event you see. You must find an offset that aligns the ENTIRE flight.
    
    **Step 1: Identify Anchor Points**
    Use the provided "Key Telemetry Events" as strong hints.
    - **Engine Run-up / RPM Check**: Look for "run up", "check", "4000" in audio near the detected High RPM telemetry event.
    - **Takeoff**: Look for "clear for takeoff", "full power", "airspeed alive" in audio near the detected Takeoff telemetry event.
    
    **Step 2: Calculate Offset for Each Anchor**
    For each anchor, calculate: Offset = Telemetry Time - Audio Time.
    
    **Step 3: Verify Consistency**
    The offsets should be roughly the same (within a few seconds). If they differ significantly, re-evaluate your matches.
    
    **Step 4: Determine Final Offset**
    Return the offset that best satisfies all anchor points.
    
    Return a JSON object with:
    - "offset_sec": The calculated offset in seconds.
    - "confidence": A score from 0.0 to 1.0.
    - "reasoning": A detailed explanation citing the specific anchor points used (e.g., "Run-up at Audio X aligns with Telemetry Y... Takeoff at Audio A aligns with Telemetry B...").
    """
    
    user_prompt = f"""
    TRANSCRIPT:
    {transcript_text}
    
    KEY TELEMETRY EVENTS (Use these as anchors):
    {key_events_text}
    
    TELEMETRY (time, alt_agl, ias, rpm, v_spd):
    {telemetry_text}
    
    Calculate the synchronization offset.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return {
            "offset_sec": float(result.get("offset_sec", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "method": "gpt-5-nano",
            "details": {
                "reasoning": result.get("reasoning", "No reasoning provided")
            }
        }
        
    except Exception as e:
        print(f"LLM Alignment error: {e}")
        return {
            "offset_sec": 0.0, 
            "confidence": 0.0, 
            "method": "llm_error",
            "details": {"error": str(e)}
        }

def calculate_offset(transcript: dict, telemetry: dict) -> dict:
    """
    Calculates the time offset between audio and telemetry.
    Now delegates to the LLM-based approach.
    """
    print("Calculating offset using LLM...")
    return calculate_offset_with_llm(transcript, telemetry)
