import os
from openai import OpenAI
from typing import List, Dict, Any

def generate_debrief(
    transcript: dict, 
    telemetry: dict, 
    segments: List[Dict[str, Any]], 
    plane_type: str = "Unknown"
) -> str:
    """
    Generates a short and sweet CFI-style debrief of the flight.
    """
    print("\n" + "=" * 60)
    print("GENERATING FLIGHT DEBRIEF")
    print("=" * 60)

    # 1. Prepare Context
    
    # Telemetry Summary (Sampled)
    data_points = telemetry.get("data", [])
    telemetry_summary_lines = []
    step = 10 # Sample every 10 seconds
    for i in range(0, len(data_points), step):
        p = data_points[i]
        line = (f"T={int(p['time_sec'])}s "
                f"Alt={int(p['alt_agl'])} "
                f"Spd={int(p['ias'])} "
                f"RPM={int(p['rpm'])} "
                f"Bank={int(p.get('roll', 0))} "
                f"VS={int(p.get('v_spd', 0))}")
        if p.get("is_ground"):
            line += " [GND]"
        telemetry_summary_lines.append(line)
    
    telemetry_summary = "\n".join(telemetry_summary_lines)

    # Transcript Text
    transcript_text = ""
    if transcript and "segments" in transcript:
        for seg in transcript["segments"]:
            transcript_text += f"[{int(seg['start'])}-{int(seg['end'])}s]: {seg['text']}\n"

    # Segments Text
    segments_text = ""
    for seg in segments:
        segments_text += f"- {seg['name']}: {seg['start_time']}-{seg['end_time']}s ({seg.get('description', '')})\n"

    # 2. Construct Prompt
    prompt = f"""You are an experienced Certified Flight Instructor (CFI).
    
Your student has just completed a flight in a {plane_type}.
Based on the flight data below, provide a SHORT and SWEET overview debrief of the entire flight.

OBJECTIVE:
- Summarize the flight's key phases and performance.
- Highlight what went well and any major areas for improvement.
- Keep it encouraging but professional.
- The tone should be conversational, like a post-flight chat in the briefing room.
- Do NOT list every single event. Focus on the big picture.
- If there are specific maneuvers (Steep Turns, Stalls, etc.), mention how they looked based on the data.

DATA:

[FLIGHT SEGMENTS]
{segments_text}

[TRANSCRIPT]
{transcript_text}

[TELEMETRY SAMPLES]
{telemetry_summary}

OUTPUT:
A single paragraph or two short paragraphs. Plain text.
"""

    # 3. Call LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: No OpenAI API Key found. Cannot generate debrief."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful flight instructor."},
                {"role": "user", "content": prompt},
            ],
        )
        
        debrief = response.choices[0].message.content.strip()
        return debrief

    except Exception as e:
        print(f"Error generating debrief: {e}")
        return "Unable to generate debrief due to an error."


def generate_segment_debrief(
    transcript: dict,
    telemetry: dict,
    target_segment: Dict[str, Any],
    prev_segment: Dict[str, Any] = None,
    next_segment: Dict[str, Any] = None,
    plane_type: str = "Unknown"
) -> str:
    """
    Generates a focused CFI-style debrief for a specific flight segment.
    Considers previous and next segments for context.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING SEGMENT DEBRIEF: {target_segment['name']}")
    print(f"{'='*60}")

    # 1. Filter telemetry data for the target segment and adjacent segments
    data_points = telemetry.get("data", [])
    start_time = target_segment["start_time"]
    end_time = target_segment["end_time"]
    
    # Get telemetry for target segment
    target_telemetry = [
        p for p in data_points
        if start_time <= p["time_sec"] <= end_time
    ]
    
    # Sample telemetry for target segment (every 2 seconds for more detail)
    telemetry_lines = []
    step = max(1, len(target_telemetry) // 30)  # Sample ~30 points max
    for i in range(0, len(target_telemetry), step):
        p = target_telemetry[i]
        line = (f"T={int(p['time_sec'])}s "
                f"Alt={int(p['alt_agl'])} "
                f"Spd={int(p['ias'])} "
                f"RPM={int(p['rpm'])} "
                f"Bank={int(p.get('roll', 0))} "
                f"Pitch={int(p.get('pitch', 0))} "
                f"VS={int(p.get('v_spd', 0))}")
        if p.get("is_ground"):
            line += " [GND]"
        telemetry_lines.append(line)
    
    telemetry_summary = "\n".join(telemetry_lines)

    # 2. Get transcript segments for target and adjacent segments
    transcript_text = ""
    if transcript and "segments" in transcript:
        for seg in transcript["segments"]:
            seg_time = seg["start"]
            if start_time <= seg_time <= end_time:
                transcript_text += f"[{int(seg['start'])}-{int(seg['end'])}s]: {seg['text']}\n"

    # 3. Build context strings for previous and next segments
    context_info = ""
    
    if prev_segment:
        context_info += f"\n[PREVIOUS SEGMENT]\n"
        context_info += f"Name: {prev_segment['name']}\n"
        context_info += f"Time: {prev_segment['start_time']}-{prev_segment['end_time']}s\n"
        context_info += f"Description: {prev_segment.get('description', 'N/A')}\n"
    
    if next_segment:
        context_info += f"\n[NEXT SEGMENT]\n"
        context_info += f"Name: {next_segment['name']}\n"
        context_info += f"Time: {next_segment['start_time']}-{next_segment['end_time']}s\n"
        context_info += f"Description: {next_segment.get('description', 'N/A')}\n"

    # 4. Construct prompt
    prompt = f"""You are an experienced Certified Flight Instructor (CFI) providing a focused debrief on a specific segment of a training flight.

AIRCRAFT: {plane_type}

TARGET SEGMENT:
Name: {target_segment['name']}
Time: {start_time}-{end_time}s
Description: {target_segment.get('description', 'N/A')}

ADJACENT SEGMENTS (for context):
{context_info}

OBJECTIVE:
- Provide a SHORT and FOCUSED analysis of the TARGET SEGMENT only.
- Analyze the student's performance during this specific phase.
- If relevant, note how this segment was influenced by the previous segment or how it set up the next segment.
- Highlight what went well and what could be improved.
- Be encouraging but professional, like a CFI debriefing a specific maneuver.
- Keep it 2-3 sentences maximum.

DATA FOR TARGET SEGMENT:

[TRANSCRIPT]
{transcript_text if transcript_text else "No audio transcript for this segment."}

[TELEMETRY]
{telemetry_summary}

OUTPUT:
A brief, focused debrief (2-3 sentences). Plain text.
"""

    # 5. Call LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: No OpenAI API Key found. Cannot generate segment debrief."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful flight instructor."},
                {"role": "user", "content": prompt},
            ],
        )
        
        debrief = response.choices[0].message.content.strip()
        return debrief

    except Exception as e:
        print(f"Error generating segment debrief: {e}")
        return "Unable to generate segment debrief due to an error."

