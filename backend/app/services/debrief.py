import os
from openai import OpenAI
from typing import List, Dict, Any

# Segment-specific focus areas for CFI debrief analysis
# Keys match segment names from segmentation.py (case-insensitive matching used)
SEGMENT_FOCUS_AREAS: Dict[str, List[str]] = {
    "TAXI": [
        "Indicated speeds are low and safe (typically under 20 knots)",
        "Centerline maintenance based on transcript callouts",
        "Proper brake usage and control inputs",
        "Correct taxiway and runway identification from communications",
        "Appropriate engine RPM for taxi operations",
    ],
    "RUNUP": [
        "Proper engine run-up procedures and mag checks",
        "Engine parameter checks (RPM, oil pressure, temperatures)",
        "Control surface checks and freedom of movement",
        "Correct cycling of prop and systems",
        "Communication with ground/tower as appropriate",
        "Checklist usage and flow",
    ],
    "TAKEOFF": [
        "Smooth power application to full throttle",
        "Airspeed alive callout and rotation speed accuracy",
        "Wings level during initial climb",
        "Proper pitch attitude for Vy or Vx climb",
        "Centerline maintenance during ground roll",
        "Timely and correct after-takeoff checklist items",
    ],
    "SUSTAINED CLIMB": [
        "Maintaining proper climb airspeed (Vy or Vx as appropriate)",
        "Steady positive rate of climb",
        "Coordinated flight with minimal bank oscillation",
        "Proper engine power settings for climb",
        "Level-off technique and altitude capture",
    ],
    "CRUISE": [
        "Altitude maintenance within standards (+/- 100 feet)",
        "Heading and course tracking",
        "Proper power settings for cruise",
        "Trim usage for hands-off flight",
        "Situational awareness from communications",
    ],
    "SUSTAINED DESCENT": [
        "Controlled descent rate (typically 500-1000 fpm)",
        "Airspeed management during descent",
        "Power adjustments for descent profile",
        "Altitude awareness and level-off preparation",
        "Configuration changes if applicable",
    ],
    "TRAFFIC PATTERN": [
        "Proper pattern altitude maintenance",
        "Correct airspeeds for each leg (downwind, base, final)",
        "Appropriate bank angles for turns (typically 20-30 degrees)",
        "Timely position reports and communications",
        "Configuration changes at appropriate points (flaps, gear)",
        "Descent initiation and glide path management",
        "Maintaining stable approach with consistent glide slope on final",
        "Wind correction and crab/slip technique if applicable",
    ],
    "LANDING": [
        "Stable approach speed and configuration",
        "Stable pitch attitude and controlled airspeed on final",
        "Managed descent rate for approach profile",
        "Runway alignment and centerline tracking",
        "Proper flare timing and pitch attitude",
        "Smooth touchdown with appropriate descent rate reduction",
        "Centerline maintenance during rollout",
        "Proper braking and speed reduction",
        "After-landing checklist items and taxi clearance",
    ],
    "SHUTDOWN": [
        "Proper engine shutdown procedures",
        "Avionics and electrical shutdown sequence",
        "Parking brake and control lock application",
        "Post-flight checklist completion",
    ],
    # Specific maneuvers
    "MANEUVERS": [
        "Entry altitude, airspeed, and heading per standards",
        "Bank angle and pitch control appropriate for maneuver",
        "Altitude maintenance during maneuver (+/- 100 feet)",
        "Coordination (slip/skid ball centered)",
        "Recovery technique and return to straight-and-level",
        "Situational awareness and clearing turns",
    ],
    "STEEP TURNS": [
        "45-degree bank angle maintenance",
        "Altitude within +/- 100 feet throughout",
        "Airspeed within +/- 10 knots",
        "Coordinated flight (no slip/skid)",
        "Proper roll-out on entry heading (+/- 10 degrees)",
        "Back pressure and power adjustments during turn",
    ],
    "SLOW FLIGHT": [
        "Airspeed at or near stall warning (typically 5-10 knots above stall)",
        "Altitude maintenance within standards",
        "Coordinated turns without stall",
        "Proper power and pitch coordination",
        "Recognition of flight control effectiveness at low speeds",
    ],
    "POWER OFF STALL": [
        "Proper entry configuration (landing or approach config)",
        "Smooth power reduction to idle",
        "Recognition of stall warning and break",
        "Prompt and correct recovery technique",
        "Minimal altitude loss during recovery",
        "Return to coordinated flight",
    ],
    "POWER ON STALL": [
        "Proper entry configuration (takeoff or departure config)",
        "Full power application and pitch up",
        "Recognition of stall warning and break",
        "Prompt and correct recovery technique",
        "Wing-level maintenance during stall",
        "Minimal altitude loss during recovery",
    ],
    "GROUND REFERENCE": [
        "Proper altitude for maneuver (typically 600-1000 AGL)",
        "Consistent ground track maintenance",
        "Wind correction angle adjustments",
        "Coordination and airspeed control",
        "Situational awareness and traffic avoidance",
    ],
    "EMERGENCY PROCEDURE": [
        "Prompt and correct initial response",
        "Appropriate checklist or memory item execution",
        "Airspeed and configuration management",
        "Communication and decision making",
        "Safe outcome or simulated emergency resolution",
    ],
}


def get_focus_areas_for_segment(segment_name: str) -> List[str]:
    """
    Get the focus areas for a given segment name.
    Uses case-insensitive matching and handles variations in naming.
    """
    # Normalize the segment name for matching
    normalized = segment_name.strip().upper()
    
    # Direct match
    if normalized in SEGMENT_FOCUS_AREAS:
        return SEGMENT_FOCUS_AREAS[normalized]
    
    # Try with underscores replaced by spaces
    with_spaces = normalized.replace("_", " ")
    if with_spaces in SEGMENT_FOCUS_AREAS:
        return SEGMENT_FOCUS_AREAS[with_spaces]
    
    # Try with spaces replaced by underscores
    with_underscores = normalized.replace(" ", "_")
    if with_underscores in SEGMENT_FOCUS_AREAS:
        return SEGMENT_FOCUS_AREAS[with_underscores]
    
    # Partial matching for maneuvers
    maneuver_keywords = ["STEEP", "STALL", "SLOW", "GROUND REFERENCE", "EMERGENCY"]
    for keyword in maneuver_keywords:
        if keyword in normalized:
            for key in SEGMENT_FOCUS_AREAS:
                if keyword in key:
                    return SEGMENT_FOCUS_AREAS[key]
    
    # Default to general MANEUVERS if it looks like a maneuver
    if any(word in normalized for word in ["TURN", "STALL", "FLIGHT", "PROCEDURE"]):
        return SEGMENT_FOCUS_AREAS.get("MANEUVERS", [])
    
    # Return empty list if no match found
    return []

def generate_debrief(
    transcript: dict, 
    telemetry: dict, 
    segments: List[Dict[str, Any]], 
    plane_type: str = "Unknown"
) -> str:
    """
    Generates a short and sweet CFI-style debrief of the flight.
    """
    print("GENERATING FLIGHT DEBRIEF")

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # 1. Prepare Context
    
    # Telemetry Summary (Sampled)
    data_points = telemetry.get("data", [])
    telemetry_summary_lines = []
    step = 10 # Sample every 10 seconds
    for i in range(0, len(data_points), step):
        p = data_points[i]
        t_str = format_time(p['time_sec'])
        line = (f"T={t_str} "
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
            start_str = format_time(seg['start'])
            end_str = format_time(seg['end'])
            transcript_text += f"[{start_str}-{end_str}]: {seg['text']}\n"

    # Segments Text
    segments_text = ""
    for seg in segments:
        s_time = format_time(seg['start_time'])
        e_time = format_time(seg['end_time'])
        segments_text += f"- {seg['name']}: {s_time}-{e_time} ({seg.get('description', '')})\n"

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
        - When mentioning time, ALWAYS use the format MM:SS (e.g. 05:30) used in the data.
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
                {"role": "system", "content": "You are a helpful flight instructor, specializing in flight analysis and debriefing."},
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
    print(f"GENERATING SEGMENT DEBRIEF: {target_segment['name']}")

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

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
        t_str = format_time(p['time_sec'])
        line = (f"T={t_str} "
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
                s_str = format_time(seg['start'])
                e_str = format_time(seg['end'])
                transcript_text += f"[{s_str}-{e_str}]: {seg['text']}\n"

    # 3. Build context strings for previous and next segments
    context_info = ""
    
    if prev_segment:
        context_info += f"\n[PREVIOUS SEGMENT]\n"
        context_info += f"Name: {prev_segment['name']}\n"
        context_info += f"Time: {format_time(prev_segment['start_time'])}-{format_time(prev_segment['end_time'])}\n"
        context_info += f"Description: {prev_segment.get('description', 'N/A')}\n"
    
    if next_segment:
        context_info += f"\n[NEXT SEGMENT]\n"
        context_info += f"Name: {next_segment['name']}\n"
        context_info += f"Time: {format_time(next_segment['start_time'])}-{format_time(next_segment['end_time'])}\n"
        context_info += f"Description: {next_segment.get('description', 'N/A')}\n"

    # 4. Get segment-specific focus areas
    segment_name = target_segment['name']
    focus_areas = get_focus_areas_for_segment(segment_name)
    
    focus_areas_text = ""
    if focus_areas:
        focus_areas_text = "\n\nKEY AREAS TO ANALYZE FOR THIS SEGMENT:\n"
        for i, area in enumerate(focus_areas, 1):
            focus_areas_text += f"  {i}. {area}\n"
        focus_areas_text += "\nFocus your analysis on these specific aspects when reviewing the telemetry and transcript data."
    else:
        focus_areas_text = "\n\nAnalyze general flight performance for this segment."

    # 5. Construct prompt
    prompt = f"""You are an experienced Certified Flight Instructor (CFI) providing a focused debrief on a specific segment of a training flight.
        AIRCRAFT: {plane_type}

        TARGET SEGMENT:
        Name: {target_segment['name']}
        Time: {format_time(start_time)}-{format_time(end_time)}
        Description: {target_segment.get('description', 'N/A')}

        ADJACENT SEGMENTS (for context):
        {context_info}
        {focus_areas_text}

        OBJECTIVE:
        - Provide a SHORT and FOCUSED analysis of the TARGET SEGMENT only.
        - Analyze the student's performance during this specific phase, paying special attention to the key areas listed above.
        - Use the telemetry data to support your observations (e.g., "Your bank angle held steady at 45 degrees" or "Speed dropped to 65 knots during the flare").
        - If relevant, note how this segment was influenced by the previous segment or how it set up the next segment.
        - Highlight what went well and what could be improved.
        - Be encouraging but professional, like a CFI debriefing a specific maneuver.
        - When mentioning time, ALWAYS use the format MM:SS (e.g. 05:30).
        - Keep it detailed and effective, 6 sentences maximum. We want good details and analysis.

        DATA FOR TARGET SEGMENT:

        [TRANSCRIPT]
        {transcript_text if transcript_text else "No audio transcript for this segment."}

        [TELEMETRY]
        {telemetry_summary}

        OUTPUT:
        A brief, focused debrief (3-4 sentences). Plain text.
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

