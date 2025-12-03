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

def detect_landing_events(telemetry: dict, profile: dict = None) -> list:
    """
    Detects landing events throughout the flight.
    Landing criteria: Descending to ground (AGL decreasing to < 50ft), 
    lower RPMs (approach power), and potentially flaps extended.
    
    Returns a list of dicts with landing candidate info including time windows.
    """
    if not telemetry or "data" not in telemetry:
        return []
    
    data = telemetry["data"]
    if len(data) < 30:
        return []
    
    # Get thresholds from profile
    approach_rpm_max = 3500  # Default
    if profile:
        rpm_profiles = profile.get("rpm_profiles", {})
        if "approach_descent" in rpm_profiles:
            approach_rpm_max = rpm_profiles["approach_descent"][1]
    
    landing_candidates = []
    
    # Look for descent to ground with appropriate power setting
    for i in range(20, len(data) - 10):
        point = data[i]
        prev_20 = data[i-20]  # 20 seconds before
        future_10 = data[min(i+10, len(data)-1)]  # 10 seconds after
        
        alt_agl = point.get("alt_agl", 0)
        prev_alt = prev_20.get("alt_agl", 0)
        rpm = point.get("rpm", 0)
        v_spd = point.get("v_spd", 0)
        flaps = point.get("flaps", 0)
        gnd_spd = point.get("gnd_spd", 0)
        
        # Landing criteria:
        # 1. Was in the air (prev_alt > 100ft) and now descending to ground (alt_agl < 50ft)
        # 2. Descending (negative v_spd)
        # 3. Lower RPM (approach power range)
        # 4. Optionally flaps extended (flaps > 0)
        
        if (prev_alt > 100 and alt_agl < 50 and 
            v_spd < 0 and 
            rpm < approach_rpm_max and rpm > 1000):  # Some power, not idle
            
            # Check if we actually touch down (speed decreases significantly after)
            if future_10.get("gnd_spd", 0) < gnd_spd * 0.7:  # Speed drops
                landing_candidates.append({
                    "time_sec": point["time_sec"],
                    "alt_agl": alt_agl,
                    "rpm": rpm,
                    "v_spd": v_spd,
                    "flaps": flaps,
                    "gnd_spd": gnd_spd
                })
    
    # Filter to distinct landings (at least 60 seconds apart)
    distinct_landings = []
    for candidate in landing_candidates:
        if not distinct_landings or (candidate["time_sec"] - distinct_landings[-1]["time_sec"]) > 60:
            distinct_landings.append(candidate)
    
    return distinct_landings


def detect_rpm_transitions(telemetry: dict, profile: dict = None) -> list:
    """
    Detects significant RPM transitions from high to low (power reductions to idle).
    These often correspond to:
    - End of run-up checks (high RPM -> idle)
    - Final approach (reducing to approach power or idle)
    
    Returns a list of dicts with transition info.
    """
    if not telemetry or "data" not in telemetry:
        return []
    
    data = telemetry["data"]
    if len(data) < 20:
        return []
    
    # Get thresholds from profile
    high_rpm_threshold = 3500  # Default for "high power"
    idle_rpm_threshold = 2200  # Default for "idle/low power"
    
    if profile:
        rpm_profiles = profile.get("rpm_profiles", {})
        maneuvers = profile.get("maneuver_thresholds", {})
        
        # High power: use run-up min or slow flight min
        if "rpm_runup_min" in maneuvers:
            high_rpm_threshold = maneuvers["rpm_runup_min"]
        elif "run_up" in rpm_profiles:
            high_rpm_threshold = rpm_profiles["run_up"][0]
        
        # Idle: use upper bound of idle/taxi range
        if "idle_taxi" in rpm_profiles:
            idle_rpm_threshold = rpm_profiles["idle_taxi"][1]
    
    transitions = []
    
    # Look for transitions from high RPM to low RPM
    for i in range(10, len(data) - 5):
        point = data[i]
        prev_10 = data[i-10]  # 10 seconds before
        future_5 = data[min(i+5, len(data)-1)]  # 5 seconds after
        
        rpm_prev = prev_10.get("rpm", 0)
        rpm_current = point.get("rpm", 0)
        rpm_future = future_5.get("rpm", 0)
        
        # Detect transition: was high, now dropping, stays low
        if (rpm_prev > high_rpm_threshold and 
            rpm_current < high_rpm_threshold and 
            rpm_future < idle_rpm_threshold):
            
            # Calculate the rate of RPM drop
            rpm_drop = rpm_prev - rpm_current
            
            transitions.append({
                "time_sec": point["time_sec"],
                "rpm_before": rpm_prev,
                "rpm_after": rpm_current,
                "rpm_drop": rpm_drop,
                "alt_agl": point.get("alt_agl", 0),
                "gnd_spd": point.get("gnd_spd", 0)
            })
    
    # Filter to distinct transitions (at least 30 seconds apart)
    distinct_transitions = []
    for transition in transitions:
        if not distinct_transitions or (transition["time_sec"] - distinct_transitions[-1]["time_sec"]) > 30:
            distinct_transitions.append(transition)
    
    return distinct_transitions


def detect_key_telemetry_events(telemetry: dict, profile: dict = None) -> str:
    """
    Analyzes telemetry to find key events like Run-ups, Takeoffs, and Landings.
    Returns a formatted string description.
    """
    if not telemetry or "data" not in telemetry:
        return "No telemetry data available."
    
    # Defaults
    runup_rpm_min = 2000
    takeoff_speed_min = 40
    
    if profile:
        # Extract thresholds from profile
        maneuvers = profile.get("maneuver_thresholds", {})
        rpm_profiles = profile.get("rpm_profiles", {})
        
        # Use specific run-up min if available, or lower bound of run-up range
        if "rpm_runup_min" in maneuvers:
            runup_rpm_min = maneuvers["rpm_runup_min"]
        elif "run_up" in rpm_profiles:
            runup_rpm_min = rpm_profiles["run_up"][0]
            
        # Use V_S0 or similar as a proxy for takeoff roll start speed check
        v_speeds = profile.get("v_speeds", {})
        if "v_s0" in v_speeds:
            takeoff_speed_min = v_speeds["v_s0"]
    
    data = telemetry["data"]
    events = []
    
    # 1. Detect Run-up (High RPM, Low Speed, On Ground)
    # Criteria: RPM > 2000, Speed < 5 kts, Alt AGL < 50 ft
    run_up_candidates = []
    for point in data:
        rpm = point.get("rpm", 0)
        speed = point.get("ias", 0) # Indicated Airspeed
        gnd_spd = point.get("gnd_spd", 0) # Ground Speed
        alt = point.get("alt_agl", 0)
        
        # Use ground speed if available and low, otherwise IAS might be noisy at 0
        # But IAS is what "airspeed alive" refers to. For run-up, we want to be stationary.
        is_stationary = gnd_spd < 5 if gnd_spd is not None else speed < 10
        
        if rpm > runup_rpm_min and is_stationary and alt < 50:
            run_up_candidates.append(point["time_sec"])
            
    if run_up_candidates:
        # Group consecutive seconds
        start = run_up_candidates[0]
        end = run_up_candidates[0]
        distinct_runups = []
        
        for t in run_up_candidates[1:]:
            if t - end < 5: # continuous event
                end = t
            else:
                if end - start > 5: # Minimum duration 5s for a run-up
                    distinct_runups.append((start, end))
                start = t
                end = t
        if end - start > 5:
            distinct_runups.append((start, end))
            
        for start, end in distinct_runups:
            events.append(f"- Potential Engine Run-up (RPM > {runup_rpm_min}, Stationary): {start:.1f}s to {end:.1f}s")

    # 2. Detect Takeoff (Full Power, Speed Increasing, then Climb)
    # Look for transition from Ground to Air
    # Simple heuristic: Speed goes from < 30 to > 50 and stays high, VSpd becomes positive
    
    takeoff_candidates = []
    # Ensure we have enough data points
    if len(data) > 20:
        for i in range(10, len(data)-10):
            point = data[i]
            prev = data[i-10]
            
            # Check for start of takeoff roll: Speed low -> increasing
            if point.get("ias", 0) > takeoff_speed_min and prev.get("ias", 0) < (takeoff_speed_min - 10):
                # Check if it continues to accelerate or climb (look ahead)
                # Simple check: is speed at i+10 also high?
                future = data[min(i+10, len(data)-1)]
                if future.get("ias", 0) > 45:
                     takeoff_candidates.append(point["time_sec"])
    
    # Filter takeoff candidates to find the first distinct one
    if takeoff_candidates:
        # Just take the first one that looks like a takeoff
        t = takeoff_candidates[0]
        events.append(f"- Potential Takeoff Roll (Speed increasing past {takeoff_speed_min}kt): ~{t:.1f}s")

    # 3. Detect Landings
    landings = detect_landing_events(telemetry, profile)
    for landing in landings:
        events.append(f"- Potential Landing (Descent to ground, RPM {landing['rpm']:.0f}, Flaps {landing['flaps']:.0f}): ~{landing['time_sec']:.1f}s")

    if not events:
        return "No significant telemetry events detected automatically."
        
    return "\n".join(events)

def get_transcript_window(transcript: dict, center_time: float, buffer_sec: float = 30.0) -> str:
    """
    Extracts a window of transcript segments around a center time.
    Returns formatted transcript text for that window.
    """
    if not transcript or "segments" not in transcript:
        return ""
    
    start_time = center_time - buffer_sec
    end_time = center_time + buffer_sec
    
    lines = []
    for segment in transcript["segments"]:
        if start_time <= segment["start"] <= end_time:
            lines.append(f"[{segment['start']:.1f}s] {segment['text'].strip()}")
    
    return "\n".join(lines) if lines else "(No transcript in this window)"


def get_telemetry_window(telemetry: dict, center_time: float, buffer_sec: float = 30.0) -> str:
    """
    Extracts a window of telemetry data around a center time.
    Returns CSV-formatted telemetry for that window.
    """
    if not telemetry or "data" not in telemetry:
        return ""
    
    start_time = center_time - buffer_sec
    end_time = center_time + buffer_sec
    
    lines = ["time,alt,speed,rpm,v_spd,flaps"]
    
    for point in telemetry["data"]:
        t = point.get("time_sec", 0)
        if start_time <= t <= end_time:
            alt = f"{point.get('alt_agl', 0):.0f}"
            speed = f"{point.get('ias', 0):.0f}"
            rpm = f"{point.get('rpm', 0):.0f}"
            v_spd = f"{point.get('v_spd', 0):.0f}"
            flaps = f"{point.get('flaps', 0):.0f}"
            lines.append(f"{t:.1f},{alt},{speed},{rpm},{v_spd},{flaps}")
    
    return "\n".join(lines) if len(lines) > 1 else "(No telemetry in this window)"


def scan_transcript_for_keywords(transcript: dict, profile: dict = None) -> str:
    """
    Scans the transcript for specific aviation keywords to help with alignment.
    Returns a formatted string of found keywords and their timestamps.
    """
    if not transcript or "segments" not in transcript:
        return "No transcript data available."

    keywords = [
        "run up", "run-up", "check", "magneto", "rpm", "4000", "four thousand", # Run-up related
        "airspeed", "alive", "rotate", "takeoff", "power", "full power", "clear", # Takeoff related
        "landing", "final", "flare", "touchdown", "base", "downwind", # Landing related
        "idle", "go idle", "going to idle", "pulling back", "pull back", "power back", "reduce power" # Power reduction
    ]
    
    found_events = []
    
    for segment in transcript["segments"]:
        text = segment["text"].lower()
        start = segment["start"]
        
        # Check for keywords
        matched = [kw for kw in keywords if kw in text]
        if matched:
            found_events.append(f"[{start:.1f}s] '{segment['text'].strip()}' (Matches: {', '.join(matched)})")
            
    if not found_events:
        return "No specific alignment keywords found in transcript."
        
    return "\n".join(found_events)

def build_candidate_events(transcript: dict, telemetry: dict, profile: dict = None) -> list:
    """
    Builds a list of candidate alignment events with buffered audio and telemetry windows.
    Each candidate includes the event type, telemetry time, and context windows.
    """
    candidates = []
    
    if not telemetry or "data" not in telemetry:
        return candidates
    
    data = telemetry["data"]
    
    # Get thresholds from profile
    runup_rpm_min = 3500
    takeoff_speed_min = 40
    
    if profile:
        maneuvers = profile.get("maneuver_thresholds", {})
        rpm_profiles = profile.get("rpm_profiles", {})
        v_speeds = profile.get("v_speeds", {})
        
        if "rpm_runup_min" in maneuvers:
            runup_rpm_min = maneuvers["rpm_runup_min"]
        elif "run_up" in rpm_profiles:
            runup_rpm_min = rpm_profiles["run_up"][0]
            
        if "v_s0" in v_speeds:
            takeoff_speed_min = v_speeds["v_s0"]
    
    # 1. Find Run-up candidates
    run_up_times = []
    for point in data:
        rpm = point.get("rpm", 0)
        gnd_spd = point.get("gnd_spd", 0)
        alt = point.get("alt_agl", 0)
        
        if rpm > runup_rpm_min and gnd_spd < 5 and alt < 50:
            run_up_times.append(point["time_sec"])
    
    # Group consecutive run-ups
    if run_up_times:
        start = run_up_times[0]
        end = run_up_times[0]
        
        for t in run_up_times[1:]:
            if t - end < 5:
                end = t
            else:
                if end - start > 5:
                    # Use middle of run-up as candidate time
                    candidate_time = (start + end) / 2
                    candidates.append({
                        "type": "run_up",
                        "telemetry_time": candidate_time,
                        "description": f"Engine Run-up at {candidate_time:.1f}s (RPM > {runup_rpm_min})"
                    })
                start = t
                end = t
        
        if end - start > 5:
            candidate_time = (start + end) / 2
            candidates.append({
                "type": "run_up",
                "telemetry_time": candidate_time,
                "description": f"Engine Run-up at {candidate_time:.1f}s (RPM > {runup_rpm_min})"
            })
    
    # 2. Find Takeoff candidates
    if len(data) > 20:
        for i in range(10, len(data) - 10):
            point = data[i]
            prev = data[i-10]
            
            if point.get("ias", 0) > takeoff_speed_min and prev.get("ias", 0) < (takeoff_speed_min - 10):
                future = data[min(i+10, len(data)-1)]
                if future.get("ias", 0) > 45:
                    candidates.append({
                        "type": "takeoff",
                        "telemetry_time": point["time_sec"],
                        "description": f"Takeoff Roll at {point['time_sec']:.1f}s (Speed increasing past {takeoff_speed_min}kt)"
                    })
                    break  # Only take first takeoff
    
    # 3. Find Landing candidates
    landings = detect_landing_events(telemetry, profile)
    for landing in landings:
        candidates.append({
            "type": "landing",
            "telemetry_time": landing["time_sec"],
            "description": f"Landing at {landing['time_sec']:.1f}s (Descent to ground, RPM {landing['rpm']:.0f}, Flaps {landing['flaps']:.0f})"
        })
    
    # 4. Find RPM transitions (high to low / power reductions to idle)
    rpm_transitions = detect_rpm_transitions(telemetry, profile)
    for transition in rpm_transitions:
        # Determine context based on altitude
        context = "on ground" if transition["alt_agl"] < 50 else f"at {transition['alt_agl']:.0f}ft AGL"
        candidates.append({
            "type": "rpm_reduction",
            "telemetry_time": transition["time_sec"],
            "description": f"Power Reduction at {transition['time_sec']:.1f}s ({context}, RPM {transition['rpm_before']:.0f} -> {transition['rpm_after']:.0f})"
        })
    
    # Add buffered windows to each candidate
    for candidate in candidates:
        t = candidate["telemetry_time"]
        candidate["telemetry_window"] = get_telemetry_window(telemetry, t, buffer_sec=30.0)
        # Note: We'll match transcript windows after we find the offset candidates
    
    return candidates


def calculate_offset_with_llm(transcript: dict, telemetry: dict, profile: dict = None) -> dict:
    """
    Uses OpenAI's gpt-5-mini to determine the time offset.
    Now uses buffered candidate events for better alignment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Falling back to default offset.")
        return {"offset_sec": 0.0, "confidence": 0.0, "method": "fallback_no_key"}

    client = OpenAI(api_key=api_key)
    
    # Build candidate events with telemetry windows
    candidates = build_candidate_events(transcript, telemetry, profile)
    
    if not candidates:
        print("No candidate events found for alignment.")
        return {"offset_sec": 0.0, "confidence": 0.0, "method": "no_candidates"}
    
    # Get full transcript and keyword matches for context
    key_transcript_events_text = scan_transcript_for_keywords(transcript, profile)
    
    # Build candidate descriptions with windows
    candidate_descriptions = []
    for i, candidate in enumerate(candidates, 1):
        desc = f"\n**CANDIDATE {i}: {candidate['type'].upper()}**\n"
        desc += f"Description: {candidate['description']}\n"
        desc += f"\nTelemetry Window (±30s around event):\n{candidate['telemetry_window']}\n"
        candidate_descriptions.append(desc)
    
    candidates_text = "\n".join(candidate_descriptions)
    
    # Get a sample of full transcript for reference
    transcript_text = preprocess_transcript(transcript)
    if len(transcript_text) > 30000:
        transcript_text = transcript_text[:30000] + "...(truncated)"

    system_prompt = """
You are an expert flight data analyst. Your task is to synchronize an audio transcript with flight telemetry data.

The audio transcript has timestamps relative to the start of the audio recording.
The telemetry data has timestamps relative to the start of the telemetry recording.

There is a time offset between them: **Telemetry Time = Audio Time + Offset**

**YOUR TASK:**
You will be given several CANDIDATE EVENTS detected from telemetry (run-ups, takeoffs, landings, power reductions), each with:
- A telemetry time window showing flight data around that event
- A description of what the telemetry shows

You will also be given:
- The full audio transcript with timestamps
- Keywords found in the transcript that might match these events

**INSTRUCTIONS:**

1. **For each candidate event**, look through the transcript to find where pilots might be discussing or performing that event.
   - For RUN-UPS: Look for "run up", "check", "magneto", "4000 RPM", etc.
   - For TAKEOFFS: Look for "clear for takeoff", "full power", "airspeed alive", "rotate", etc.
   - For LANDINGS: Look for "landing", "final", "flare", "touchdown", "base", "downwind", etc.
   - For POWER REDUCTIONS (RPM drops): Look for "idle", "go idle", "going to idle", "pulling back", "pull back", "power back", "reduce power"
     * IMPORTANT: When matching power reductions, the "idle" callout should occur RIGHT BEFORE or AT THE SAME TIME as the RPM starts to drop in telemetry
     * This helps identify the end of run-ups (after checking magnetos at high RPM, pilot says "going to idle")
     * Also helps identify final approach phases (reducing power for landing)

2. **Match telemetry patterns to transcript timing**:
   - If telemetry shows a run-up at 500s, and transcript mentions "run up" at 450s, the offset is ~50s
   - If telemetry shows RPM dropping at 600s, and transcript mentions "going to idle" at 550s, the offset is ~50s
   - Calculate: Offset = Telemetry Time - Audio Time

3. **Verify consistency across multiple events**:
   - Calculate offset for each candidate you can match
   - The offsets should be similar (within a few seconds)
   - If they differ significantly, re-evaluate or choose the most reliable match

4. **Prioritize critical flight phases**:
   - Takeoffs and landings are usually more precisely timed in pilot communications
   - Power reductions to idle are very precise timing events (pilot says "idle" and immediately reduces power)
   - Run-ups can be more variable in timing

5. **Return your analysis** as a JSON object with:
   - "offset_sec": The calculated offset in seconds (Telemetry Time - Audio Time)
   - "confidence": Score from 0.0 to 1.0 based on how well events align
   - "reasoning": Detailed explanation citing specific matches (e.g., "Run-up at telemetry 500s matches transcript 'run up check' at 450s, giving offset of 50s. Power reduction at telemetry 600s matches 'going to idle' at 550s, giving offset of 50s. Consistent offset of 50s across events.")

**IMPORTANT**: Look for MULTIPLE anchor points and verify they give consistent offsets. Don't just use the first match you find. Power reduction events are especially valuable for precise alignment.
"""
    
    user_prompt = f"""
FULL TRANSCRIPT (for reference):
{transcript_text}

KEY TRANSCRIPT EVENTS (Keywords found):
{key_transcript_events_text}

CANDIDATE EVENTS FROM TELEMETRY (with buffered windows):
{candidates_text}

Analyze these candidates and calculate the synchronization offset. Look for matches between the telemetry events and transcript mentions, verify consistency across multiple events, and return your analysis.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
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
            "method": "gpt-5-mini",
            "details": {
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "candidates_used": len(candidates)
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

def calculate_offset(transcript: dict, telemetry: dict, profile: dict = None) -> dict:
    """
    Calculates the time offset between audio and telemetry.
    Now delegates to the LLM-based approach.
    """
    print("Calculating offset using LLM...")
    return calculate_offset_with_llm(transcript, telemetry, profile)
