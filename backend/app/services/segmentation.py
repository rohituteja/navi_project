import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ============================================================================
# ENHANCED FLIGHT PHASE DEFINITIONS
# ============================================================================


class FlightPhase(Enum):
    """Well-defined flight phases with clear boundaries"""

    TAXI = "TAXI"
    RUNUP = "RUNUP"    
    TAKEOFF = "TAKEOFF"
    SUSTAINED_CLIMB = "SUSTAINED CLIMB"
    CRUISE = "CRUISE"
    SUSTAINED_DESCENT = "SUSTAINED DESCENT"
    DOWNWIND = "DOWNWIND"
    BASE = "BASE"
    FINAL = "FINAL"
    LANDING = "LANDING"
    TOUCH_AND_GO = "TOUCH AND GO"
    SHUTDOWN = "SHUTDOWN"

    # Maneuvers
    MANEUVERS = "MANEUVERS"
    STEEP_TURNS = "STEEP TURNS"
    SLOW_FLIGHT = "SLOW FLIGHT"
    POWER_OFF_STALL = "POWER OFF STALL"
    POWER_ON_STALL = "POWER ON STALL"
    GROUND_REFERENCE = "GROUND REFERENCE"
    EMERGENCY_PROCEDURE = "EMERGENCY PROCEDURE"


@dataclass
class PhaseSignature:
    """Telemetry signature for a flight phase"""

    alt_agl_range: Tuple[float, float]  # (min, max) feet
    ias_range: Tuple[float, float]  # (min, max) knots
    rpm_range: Tuple[float, float]  # (min, max)
    vspd_range: Tuple[float, float]  # (min, max) fpm
    duration_range: Tuple[float, float]  # (min, max) seconds
    ground_speed_max: Optional[float]  # Max GS for ground ops
    keywords: List[str]  # Audio transcript keywords


# Define signatures for each phase
PHASE_SIGNATURES = {
    FlightPhase.TAXI: PhaseSignature(
        alt_agl_range=(0, 10),
        ias_range=(0, 25),
        rpm_range=(800, 1500),
        vspd_range=(-50, 50),
        duration_range=(30, 600),
        ground_speed_max=25,
        keywords=["taxi", "alpha", "bravo", "charlie", "hold short", "tower", "ground", "ramp"],
    ),
    FlightPhase.RUNUP: PhaseSignature(
        alt_agl_range=(0, 5),
        ias_range=(0, 5),
        rpm_range=(1500, 2300),
        vspd_range=(-50, 50),
        duration_range=(60, 300),
        ground_speed_max=0,
        keywords=[
            "runup",
            "run-up",
            "mag",
            "magneto",
            "left mag",
            "right mag",
            "controls",
        ],
    ),
    FlightPhase.STEEP_TURNS: PhaseSignature(
        alt_agl_range=(1000, 8000),
        ias_range=(80, 120),
        rpm_range=(2000, 2500),
        vspd_range=(-500, 500),
        duration_range=(60, 300),
        ground_speed_max=None,
        keywords=["steep turn", "45 degree", "bank", "clearing turn"],
    ),
    FlightPhase.SLOW_FLIGHT: PhaseSignature(
        alt_agl_range=(1500, 8000),
        ias_range=(40, 65),
        rpm_range=(1500, 2300),
        vspd_range=(-200, 200),
        duration_range=(60, 300),
        ground_speed_max=None,
        keywords=[
            "slow flight",
            "minimum controllable",
            "full flaps",
            "just above stall",
        ],
    ),
    FlightPhase.POWER_OFF_STALL: PhaseSignature(
        alt_agl_range=(2000, 8000),
        ias_range=(35, 70),
        rpm_range=(800, 1800),
        vspd_range=(-1000, 500),
        duration_range=(30, 180),
        ground_speed_max=None,
        keywords=[
            "power off stall",
            "landing configuration",
            "idle power",
            "stall horn",
        ],
    ),
}

# ============================================================================
# CANDIDATE GENERATOR (HEURISTIC SEGMENTATION)
# ============================================================================

def smooth_classifications(classifications, data_points, min_duration):
    """Eliminate very short segments (likely noise) by merging them into neighbors."""
    if not classifications:
        return []
        
    smoothed = classifications.copy()
    i = 0
    
    while i < len(smoothed):
        # 1. Identify continuous segment
        start = i
        current_class = smoothed[i]
        while i < len(smoothed) and smoothed[i] == current_class:
            i += 1
        end = i
        
        # 2. Check duration
        if start < len(data_points) and end-1 < len(data_points):
            t_start = data_points[start]["time_sec"]
            t_end = data_points[end-1]["time_sec"]
            duration = t_end - t_start
            
            # 3. Merge if too short
            if duration < min_duration:
                # Determine neighbor to merge with (prefer previous to extend current state)
                prev_class = smoothed[start - 1] if start > 0 else None
                next_class = smoothed[end] if end < len(smoothed) else None
                
                fill_class = prev_class if prev_class else next_class
                
                if fill_class:
                    for j in range(start, end):
                        smoothed[j] = fill_class
                        
    return smoothed

def generate_candidates(telemetry: dict, profile: dict = None) -> List[Dict[str, Any]]:
    """
    Slices the flight into 'Regions of Interest' (ROI) based on physical state changes.
    """
    data_points = telemetry.get("data", [])
    if not data_points:
        return []

    candidates = []
    
    # Extract thresholds from profile
    high_rpm_threshold = 1700
    steep_turn_roll_min = 30
    
    if profile:
        rpm_profiles = profile.get("rpm_profiles", {})
        maneuvers = profile.get("maneuver_thresholds", {})
        
        if "idle_taxi" in rpm_profiles:
            high_rpm_threshold = rpm_profiles["idle_taxi"][1]
        
        if "steep_turn_roll_min" in maneuvers:
            steep_turn_roll_min = maneuvers["steep_turn_roll_min"]
    
    # 1. Point-wise classification
    point_classifications = []
    for i, p in enumerate(data_points):
        label = "UNKNOWN"
        is_ground = p.get("is_ground", False)
        
        if is_ground:
            if p.get("rpm", 0) > high_rpm_threshold:
                label = "GROUND_HIGH_RPM" 
            elif p.get("gnd_spd", 0) > 3: # Moving > 3kts
                label = "TAXI"
            else:
                label = "STATIONARY"
        else:
            roll_abs = abs(p.get("roll", 0))
            pitch_abs = abs(p.get("pitch", 0))
            v_spd = p.get("v_spd", 0)
            
            if roll_abs > steep_turn_roll_min:
                label = "MANEUVER_HIGH_BANK"
            elif pitch_abs > 15:
                label = "MANEUVER_HIGH_PITCH"
            elif v_spd > 500:
                label = "CLIMB"
            elif v_spd < -500:
                label = "DESCENT"
            else:
                label = "STABLE_FLIGHT"
                
        point_classifications.append(label)
        
    # 2. Temporal Smoothing
    MIN_SEGMENT_DURATION = 5 # seconds
    smoothed = smooth_classifications(point_classifications, data_points, MIN_SEGMENT_DURATION)
    
    # 3. Merge consecutive identical labels
    if not smoothed:
        return []
        
    current_label = smoothed[0]
    start_idx = 0
    
    for i in range(1, len(smoothed)):
        label = smoothed[i]
        if label != current_label:
            end_idx = i - 1
            candidates.append({
                "name": current_label,
                "start_time": int(data_points[start_idx]["time_sec"]),
                "end_time": int(data_points[end_idx]["time_sec"]),
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            current_label = label
            start_idx = i
            
    # Add last segment
    candidates.append({
        "name": current_label,
        "start_time": int(data_points[start_idx]["time_sec"]),
        "end_time": int(data_points[-1]["time_sec"]),
        "start_idx": start_idx,
        "end_idx": len(data_points) - 1
    })
    
    return candidates

# ============================================================================
# ENHANCED LLM INTEGRATION
# ============================================================================

def create_enhanced_prompt(
    telemetry_summary: str,
    transcript_text: str,
    candidate_segments: List[Dict[str, Any]],
    plane_type: str = "Unknown",
    profile: dict = None
) -> str:
    candidate_text = ""
    if candidate_segments:
        candidate_text = "\n\nCANDIDATE SEGMENTS (Heuristic Regions of Interest):\n"
        for seg in candidate_segments:
            # Only show segments > 5 seconds to reduce noise in prompt
            if seg["end_time"] - seg["start_time"] > 5:
                candidate_text += (
                    f"  - {seg['name']}: {seg['start_time']}-{seg['end_time']}s\n"
                )

    profile_text = ""
    if profile:
        profile_text = f"\n\nAIRCRAFT PROFILE ({plane_type}):\n"
        profile_text += json.dumps(profile, indent=2)

    prompt = f"""You are an expert flight instructor analyzing a flight training lesson. Your task is to identify and label specific flight segments for the entire flight, based on the provided telemetry and audio transcript data.
        OBJECTIVE:
        Create a chronological timeline of the flight phases using a STRICT STATE MACHINE approach.
        Time values in the JSON output MUST be in MM:SS format (e.g., 00:00, 01:30, 05:45).

        AIRCRAFT CONTEXT:
        Aircraft Type: {plane_type}
        Use your knowledge of this specific aircraft's operating parameters (V-speeds, RPM ranges, performance characteristics).
        {profile_text}

        INPUT DATA:

        1. TELEMETRY SUMMARY (Sampled):
        {telemetry_summary}

        2. AUDIO TRANSCRIPT:
        {transcript_text}

        3. HEURISTIC CANDIDATES (Physics-based hints):
        {candidate_text}

        ALLOWED STATES (You MUST pick from this list ONLY):
        - TAXI: Movement on the ground (Taxi out, Taxi in, Taxi to Runway). (<25kts)
        - RUNUP: Engine run-up, mag checks, cycling prop. (Ground, 0 speed, High RPM)
        - TAKEOFF: Takeoff roll and initial climb to 500' AGL. (Ground -> Air, High RPM, Accel). IMPORTANT: Include ~10 seconds BEFORE liftoff (taxi onto runway, lineup) and ~10 seconds AFTER liftoff (initial climb).
        - SUSTAINED CLIMB: Climb from 500' AGL to Cruise Altitude or Maneuver Altitude.
        - CRUISE: Level flight for transit.
        - MANEUVERS: General category for airwork (Steep Turns, Stalls, Slow Flight, etc.). *Prefer specific maneuver names like STEEP TURNS, SLOW FLIGHT, POWER OFF STALL, POWER ON STALL.*
        - SUSTAINED DESCENT: Descent from altitude to traffic pattern altitude.
        - TRAFFIC PATTERN: Operations in the airport pattern (Downwind, Base, Final). Includes landing prep and final approach.
        - LANDING: Final approach flare, touchdown, and roll-out. IMPORTANT: Include ~10 seconds BEFORE touchdown (short final, flare) and ~10 seconds AFTER touchdown.
        - SHUTDOWN: Engine shutdown and securing.

        STATE TRANSITION LOGIC (Follow this flow):
        1. TAXI -> RUNUP, TAKEOFF, SHUTDOWN
        2. RUNUP -> TAXI, TAKEOFF
        3. TAKEOFF -> SUSTAINED CLIMB or TRAFFIC PATTERN
        4. SUSTAINED CLIMB -> CRUISE, MANEUVERS, TRAFFIC PATTERN, SUSTAINED DESCENT
        5. CRUISE <-> MANEUVERS
        6. CRUISE, MANEUVERS -> SUSTAINED DESCENT, TRAFFIC PATTERN
        7. SUSTAINED DESCENT -> TRAFFIC PATTERN, LANDING
        8. TRAFFIC PATTERN -> LANDING, SUSTAINED CLIMB (for go arounds)
        9. LANDING -> TAXI, TAKEOFF (Touch & Go)

        CRITICAL RULES:
        1. **Granularity**: Distinguish between TAXI and RUNUP.
        2. **Run-up Detection**: High RPM, 0 Ground Speed.
        3. **Run-up Termination**: Ends with "runup complete" or movement.
        4. **Takeoff vs Climb**: TAKEOFF ends when established in climb (~500' AGL).
        5. **Pattern Work**: Can differ if staying in pattern.
        6. **Transcript is Key**: Use pilot calls.

        OUTPUT FORMAT:
        Return a JSON object with a "segments" list.
        Example:
        {{
        "segments": [
            {{
            "name": "TAXI",
            "start_time": 0,
            "end_time": 45,
            "description": "Engine start and avionics setup, transition to runup area",
            "confidence": 0.95
            }},
            ...
        ]
        }}
        """
    return prompt


def detect_engine_start(data: List[Dict], profile: dict = None) -> Optional[Dict[str, Any]]:
    """Detect engine start - first RPM rise above idle threshold."""
    # Get idle RPM from profile, default to 800
    idle_rpm = 800
    if profile and "rpm_profiles" in profile:
        idle_range = profile["rpm_profiles"].get("idle_taxi", [600, 1000])
        idle_rpm = idle_range[0] if isinstance(idle_range, list) else 600
    
    for i, p in enumerate(data):
        current_rpm = p.get("rpm", 0)
        prev_rpm = data[i-1].get("rpm", 0) if i > 0 else 0
        
        if current_rpm > idle_rpm and prev_rpm < idle_rpm:
            return {
                "name": "ENGINE_START",
                "time": int(p["time_sec"]),
                "evidence": f"RPM rose to {int(current_rpm)} at T={int(p['time_sec'])}s",
                "confidence": 0.95
            }
    return None


def detect_takeoffs(data: List[Dict], profile: dict = None) -> List[Dict[str, Any]]:
    """Detect takeoff events - acceleration through rotation speed on ground."""
    takeoffs = []
    
    # Get thresholds from profile
    takeoff_ias = 40
    high_rpm = 4000
    if profile:
        if "maneuver_thresholds" in profile:
            takeoff_ias = profile["maneuver_thresholds"].get("takeoff_ias_min", 40)
        if "rpm_profiles" in profile:
            # Use cruise or full power RPM as takeoff threshold
            cruise_range = profile["rpm_profiles"].get("cruise", [4500, 5200])
            high_rpm = cruise_range[0] if isinstance(cruise_range, list) else 4000
    
    for i in range(10, len(data) - 10):
        current = data[i]
        prev = data[i-5]
        
        # Accelerating through takeoff speed on ground with high RPM
        is_accelerating = current["ias"] > takeoff_ias and prev["ias"] < 30
        has_high_rpm = current.get("rpm", 0) > high_rpm
        on_ground = current.get("is_ground", False)
        
        if is_accelerating and has_high_rpm and on_ground:
            # Ensure distinct (>60s apart from previous takeoff)
            if not takeoffs or (current["time_sec"] - takeoffs[-1]["time"] > 60):
                takeoffs.append({
                    "name": "TAKEOFF",
                    "time": int(current["time_sec"]),
                    "evidence": f"Speed {int(current['ias'])}kts, RPM {int(current.get('rpm', 0))}, accelerating on ground",
                    "confidence": 0.9
                })
    
    return takeoffs


def detect_landings(data: List[Dict], profile: dict = None) -> List[Dict[str, Any]]:
    """Detect landing/touchdown events - altitude drops to ground."""
    landings = []
    
    for i in range(10, len(data) - 10):
        current = data[i]
        prev = data[i-5]
        
        # AGL drops from >20ft to <10ft (touchdown)
        was_airborne = prev["alt_agl"] > 20
        now_on_ground = current["alt_agl"] < 10
        
        if was_airborne and now_on_ground:
            # Ensure distinct (>60s apart from previous landing)
            if not landings or (current["time_sec"] - landings[-1]["time"] > 60):
                landings.append({
                    "name": "LANDING",
                    "time": int(current["time_sec"]),
                    "evidence": f"AGL dropped from {int(prev['alt_agl'])}ft to {int(current['alt_agl'])}ft",
                    "confidence": 0.85
                })
    
    return landings


def detect_engine_shutdown(data: List[Dict], profile: dict = None) -> Optional[Dict[str, Any]]:
    """Detect engine shutdown - last RPM drop below idle threshold."""
    # Get idle RPM from profile
    idle_rpm = 600
    if profile and "rpm_profiles" in profile:
        idle_range = profile["rpm_profiles"].get("idle_taxi", [600, 1000])
        idle_rpm = idle_range[0] if isinstance(idle_range, list) else 600
    
    # Scan backwards from end to find last shutdown
    for i in range(len(data) - 1, 10, -1):
        current_rpm = data[i].get("rpm", 0)
        prev_rpm = data[i-5].get("rpm", 0)
        
        if current_rpm < idle_rpm and prev_rpm > idle_rpm:
            return {
                "name": "ENGINE_SHUTDOWN",
                "time": int(data[i]["time_sec"]),
                "evidence": f"RPM dropped to {int(current_rpm)} at T={int(data[i]['time_sec'])}s",
                "confidence": 0.9
            }
    return None


def validate_events_with_llm(events: List[Dict], transcript_text: str) -> List[Dict[str, Any]]:
    """Optional LLM pass to confirm/adjust detected events using transcript (±5 seconds)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not events:
        return events
    
    client = OpenAI(api_key=api_key)
    
    events_json = json.dumps(events, indent=2)
    
    prompt = f"""Review these detected flight events and confirm or adjust their timestamps based on the transcript.
    
DETECTED EVENTS (from telemetry):
{events_json}

TRANSCRIPT:
{transcript_text}

INSTRUCTIONS:
- For each event, confirm the time is correct OR adjust by up to ±5 seconds if transcript evidence suggests a more accurate time.
- Look for keywords like "engine start", "rotation", "liftoff", "touchdown", "shutdown" near the event times.
- If the event is confirmed, keep the original time.
- Return the same events list with potentially adjusted times.
- **IMPORTANT**: The 'time' field in the JSON MUST be an integer (seconds from start), NOT a "MM:SS" string.

OUTPUT JSON:
{{
    "events": [
        {{ "name": "ENGINE_START", "time": 15, "evidence": "confirmed by transcript", "confidence": 0.95 }},
        ...
    ]
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a flight data expert. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        validated = result.get("events", events)
        print(f"LLM validated {len(validated)} events")
        return validated
    except Exception as e:
        print(f"LLM validation error: {e}, using telemetry-detected events")
        return events


def identify_key_events(
    telemetry: dict, 
    transcript_text: str = "", 
    profile: dict = None,
    use_llm_validation: bool = True
) -> List[Dict[str, Any]]:
    """
    Identify major flight anchor events from telemetry data.
    Uses physics-based detection as primary, with optional LLM validation.
    
    Returns events: ENGINE_START, TAKEOFF(s), LANDING(s), ENGINE_SHUTDOWN
    """
    data = telemetry.get("data", [])
    if not data:
        return []
    
    events = []
    
    # 1. Detect Engine Start
    engine_start = detect_engine_start(data, profile)
    if engine_start:
        events.append(engine_start)
        print(f"Detected ENGINE_START at T={engine_start['time']}s")
    
    # 2. Detect Takeoffs (can be multiple for touch-and-go)
    takeoffs = detect_takeoffs(data, profile)
    events.extend(takeoffs)
    for t in takeoffs:
        print(f"Detected TAKEOFF at T={t['time']}s")
    
    # 3. Detect Landings (can be multiple)
    landings = detect_landings(data, profile)
    events.extend(landings)
    for l in landings:
        print(f"Detected LANDING at T={l['time']}s")
    
    # 4. Detect Engine Shutdown
    shutdown = detect_engine_shutdown(data, profile)
    if shutdown:
        events.append(shutdown)
        print(f"Detected ENGINE_SHUTDOWN at T={shutdown['time']}s")
    
    # Sort by time
    events.sort(key=lambda e: e["time"])
    
    print(f"Telemetry detection found {len(events)} key events")
    
    # Optional LLM validation
    if use_llm_validation and transcript_text:
        print("Running LLM validation on key events...")
        events = validate_events_with_llm(events, transcript_text)
    
    return events

def fill_segments_between_events(key_events, candidates, total_duration):
    """
    Use heuristics (candidates) to fill in the gaps between key events.
    This is a simplified version - in reality we might want more complex logic.
    For now, we will just pass the key events to the next stage LLM as "Anchors".
    """
    # We can just return the candidates, but maybe annotate them if they overlap with key events?
    return candidates

def detect_segments(
    transcript: dict, telemetry: dict, offset_sec: float = 0, plane_type: str = "Unknown", profile: dict = None
) -> List[Dict[str, Any]]:
    """
    Two-Stage LLM Approach for Segmentation.
    """
    print("SENSOR FUSION FLIGHT SEGMENTATION")
    print(f"Aircraft: {plane_type}")

    # 1. Data Ingestion & Pre-processing
    data_points = telemetry.get("data", [])
    if not data_points:
        return []

    total_duration = int(data_points[-1]["time_sec"])
    print(f"Flight duration: {total_duration // 60}m {total_duration % 60}s")

    # 2. Heuristic Segmentation (Candidate Generation)
    print("Running Heuristic Candidate Generator...")
    candidates = generate_candidates(telemetry, profile)
    print(f"Found {len(candidates)} regions of interest.")

    # 3. Prepare Context for LLM
    # Telemetry Summary
    telemetry_summary_lines = []
    step = 10
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

    # Transcript
    transcript_text = ""
    if transcript and "segments" in transcript:
        for seg in transcript["segments"]:
            t_start = int(seg["start"] + offset_sec)
            t_end = int(seg["end"] + offset_sec)
            transcript_text += f"[{t_start}-{t_end}s]: {seg['text']}\n"

    # Stage 1: Identify Key Events (Telemetry-first with optional LLM validation)
    print("Stage 1: Identifying Key Events...")
    key_events = identify_key_events(telemetry, transcript_text, profile, use_llm_validation=True)
    print(f"Found {len(key_events)} key events.")
    
    # Stage 2: Refine Boundaries (Main Segmentation)
    print("Stage 2: Refining Segments...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API key, returning heuristics only")
        return candidates

    try:
        client = OpenAI(api_key=api_key)
        
        # Add key events to prompt
        key_events_text = json.dumps(key_events, indent=2)
        
        prompt = create_enhanced_prompt(
            telemetry_summary, transcript_text, candidates, plane_type, profile
        )
        
        # Inject key events into the prompt
        prompt += f"\n\nKEY EVENTS DETECTED (Use these as anchors):\n{key_events_text}\n"

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a flight instructor. Output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        llm_segments = result.get("segments", [])
        
        print(f"LLM proposed {len(llm_segments)} segment(s)")

        # Post-process: buffer critical segments, then fill gaps
        print("Buffering critical segments (TAKEOFF, LANDING)...")
        buffered = buffer_critical_segments(llm_segments, total_duration)
        final = post_process_segments(buffered, total_duration)
        
        print(f"Total: {len(final)} segments")

        return final

    except Exception as e:
        print(f"LLM error: {e}")
        return candidates

# ============================================================================
# SEGMENT BUFFERING
# ============================================================================

def parse_time(val: Any) -> int:
    """Helper to safely parse time which might be int, string int, or MM:SS."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        # Try MM:SS format
        if ":" in val:
            try:
                parts = val.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                pass
        # Try basic int string
        try:
            return int(val)
        except ValueError:
            pass
    return 0

BUFFER_SECONDS = 10  # Buffer to add before/after critical segments
def buffer_critical_segments(segments: List[Dict[str, Any]], total_duration: int) -> List[Dict[str, Any]]:
    """
    Expands TAKEOFF and LANDING segments by stealing time from neighbors.
    - TAKEOFF: Extend start backwards (into TAXI/RUNUP), extend end forwards (into CLIMB).
    - LANDING: Extend start backwards (into TRAFFIC_PATTERN/DESCENT), extend end forwards (into TAXI).
    
    This runs BEFORE post_process_segments so that gap-filling will fix any resulting gaps.
    """
    if not segments:
        return segments

    # Sort first to ensure proper neighbor detection
    # Use parse_time safely
    sorted_segs = sorted(segments, key=lambda x: parse_time(x.get("start_time", 0)))
    buffered = []
    
    for i, seg in enumerate(sorted_segs):
        new_seg = seg.copy()
        # Ensure times are integers (LLM may return strings)
        new_seg["start_time"] = parse_time(seg.get("start_time", 0))
        new_seg["end_time"] = parse_time(seg.get("end_time", 0))
        name = seg.get("name", "").upper()

        if name == "TAKEOFF":
            # Extend start backwards
            original_start = new_seg["start_time"]
            new_start = max(0, original_start - BUFFER_SECONDS)
            new_seg["start_time"] = new_start
            
            # Extend end forward
            original_end = new_seg["end_time"]
            # Look ahead safely
            max_end = total_duration
            if i + 1 < len(sorted_segs):
                next_start = parse_time(sorted_segs[i + 1].get("start_time", 0))
                max_end = next_start - 5
            
            new_seg["end_time"] = min(original_end + BUFFER_SECONDS, max_end)
            
            print(f"Buffered TAKEOFF: {original_start}-{original_end}s -> {new_seg['start_time']}-{new_seg['end_time']}s")

        elif name == "LANDING":
            # Extend start backwards
            original_start = new_seg["start_time"]
            min_start = 0 
            if i > 0:
                 # Look back safely
                 prev_start = parse_time(sorted_segs[i - 1].get("start_time", 0))
                 # Actually we care about previous end time probably, but strict state machine implies they touch
                 # Let's just use start_time + 5 as a constraint for now? 
                 # Or just use the previous segment start + 5
                 min_start = prev_start + 5

            new_seg["start_time"] = max(min_start, original_start - BUFFER_SECONDS)
            
            # Extend end forward
            original_end = new_seg["end_time"]
            max_end = total_duration
            if i + 1 < len(sorted_segs):
                 next_start = parse_time(sorted_segs[i + 1].get("start_time", 0))
                 max_end = next_start - 5

            new_seg["end_time"] = min(original_end + BUFFER_SECONDS, max_end)
            
            print(f"Buffered LANDING: {original_start}-{original_end}s -> {new_seg['start_time']}-{new_seg['end_time']}s")

        buffered.append(new_seg)

    return buffered


def post_process_segments(segments: List[Dict[str, Any]], total_duration: int) -> List[Dict[str, Any]]:
    """
    1. Sorts segments by start time.
    2. Merges consecutive segments with the same name.
    3. Fills gaps between segments.
    """
    if not segments:
        return []
        
    # 1. Sort (ensure times are integers first)
    for seg in segments:
        seg["start_time"] = int(seg.get("start_time", 0))
        seg["end_time"] = int(seg.get("end_time", 0))
    sorted_segments = sorted(segments, key=lambda x: x["start_time"])
    
    # 2. Merge consecutive identical segments
    merged = []
    if sorted_segments:
        current = sorted_segments[0].copy()
        for next_seg in sorted_segments[1:]:
            if current["name"] == next_seg["name"]:
                # Merge
                current["end_time"] = max(current["end_time"], next_seg["end_time"])
                # Combine descriptions if they are different? For now, keep the first one or longest.
                if len(next_seg.get("description", "")) > len(current.get("description", "")):
                    current["description"] = next_seg["description"]
                
                # Merge confidence: take the weighted average or just max?
                # Let's take the max to be optimistic, or average.
                # If one has high confidence and other low, maybe the merged one is somewhere in between?
                # Let's use max for now.
                current["confidence"] = max(current.get("confidence", 0.0), next_seg.get("confidence", 0.0))
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
    
    # 3. Fill gaps
    # We will extend the end_time of the previous segment to the start_time of the next
    for i in range(len(merged) - 1):
        current = merged[i]
        next_seg = merged[i+1]
        
        if current["end_time"] < next_seg["start_time"]:
            # Gap detected
            # Extend current to meet next
            current["end_time"] = next_seg["start_time"]
            
        elif current["end_time"] > next_seg["start_time"]:
            # Overlap detected (shouldn't happen with sorted/merged but good to handle)
            # Adjust current end to match next start
            current["end_time"] = next_seg["start_time"]
            
    # Ensure start is 0 and end is total_duration
    if merged:
        merged[0]["start_time"] = 0
        merged[-1]["end_time"] = max(merged[-1]["end_time"], total_duration)
        
        # Validate continuity
        for i in range(len(merged) - 1):
            if merged[i]["end_time"] != merged[i + 1]["start_time"]:
                print(f"Warning: Gap/overlap between {merged[i]['name']} and {merged[i + 1]['name']}")
        print(f"Timeline validated: {merged[0]['start_time']}s to {merged[-1]['end_time']}s (continuous)")

    return merged
