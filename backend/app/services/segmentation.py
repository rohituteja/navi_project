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

    PREFLIGHT = "PREFLIGHT"
    TAXI_OUT = "TAXI_OUT"
    RUNUP = "RUNUP"
    TAXI_TO_RUNWAY = "TAXI_TO_RUNWAY"
    TAKEOFF = "TAKEOFF"
    SUSTAINED_CLIMB = "SUSTAINED_CLIMB"
    CRUISE = "CRUISE"
    SUSTAINED_DESCENT = "SUSTAINED_DESCENT"
    APPROACH = "APPROACH"
    DOWNWIND = "DOWNWIND"
    BASE = "BASE"
    FINAL = "FINAL"
    LANDING = "LANDING"
    TOUCH_AND_GO = "TOUCH_AND_GO"
    TAXI_IN = "TAXI_IN"
    SHUTDOWN = "SHUTDOWN"

    # Maneuvers
    STEEP_TURNS = "STEEP_TURNS"
    SLOW_FLIGHT = "SLOW_FLIGHT"
    POWER_OFF_STALL = "POWER_OFF_STALL"
    POWER_ON_STALL = "POWER_ON_STALL"
    GROUND_REFERENCE = "GROUND_REFERENCE"
    EMERGENCY_PROCEDURE = "EMERGENCY_PROCEDURE"


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
    FlightPhase.PREFLIGHT: PhaseSignature(
        alt_agl_range=(0, 5),
        ias_range=(0, 5),
        rpm_range=(0, 1200),
        vspd_range=(-50, 50),
        duration_range=(30, 600),
        ground_speed_max=0,
        keywords=["start", "oil pressure", "gauges", "preflight", "checklist"],
    ),
    FlightPhase.TAXI_OUT: PhaseSignature(
        alt_agl_range=(0, 10),
        ias_range=(0, 25),
        rpm_range=(800, 1500),
        vspd_range=(-50, 50),
        duration_range=(30, 600),
        ground_speed_max=25,
        keywords=["taxi", "alpha", "bravo", "charlie", "hold short", "tower"],
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
# IMPROVED RULE-BASED DETECTION
# ============================================================================


# ============================================================================
# CANDIDATE GENERATOR (HEURISTIC SEGMENTATION)
# ============================================================================

def smooth_classifications(classifications, data_points, min_duration):
    """Eliminate very short segments (likely noise)."""
    if not classifications:
        return []
        
    smoothed = classifications.copy()
    
    i = 0
    while i < len(smoothed):
        # Find run of same classification
        start = i
        current_class = smoothed[i]
        
        while i < len(smoothed) and smoothed[i] == current_class:
            i += 1
        
        end = i
        # Calculate duration
        # Handle edge case where data_points might be shorter than classifications if something went wrong, 
        # but they should match.
        if start < len(data_points) and end-1 < len(data_points):
            duration = data_points[end-1]["time_sec"] - data_points[start]["time_sec"]
            
            # If segment too short, merge with neighbors
            if duration < min_duration and start > 0 and end < len(smoothed):
                # Merge with longer neighbor or just previous?
                # Simple approach: merge with previous
                prev_class = smoothed[start - 1] if start > 0 else None
                next_class = smoothed[end] if end < len(smoothed) else None
                
                # Prefer merging with "STABLE_FLIGHT" or "TAXI" over "MANEUVER" if ambiguous?
                # Or just use previous.
                fill_class = prev_class or next_class
                
                if fill_class:
                    for j in range(start, end):
                        smoothed[j] = fill_class
                    
                    # Backtrack to re-evaluate merged section
                    i = start 
                    # But be careful of infinite loops if we just merged with previous and it's still short?
                    # The previous segment grows, so it should be fine.
                    # To be safe, we can just continue and let the next pass (if we did multiple) handle it,
                    # or just accept it grows.
                    # Actually, if we merge with previous, we should re-check from start of previous?
                    # For simplicity, let's just continue, as we are iterating forward.
                    # If we merge with previous, the current block becomes previous class.
                    # The loop continues from 'end' usually, but here we set i=start?
                    # If we set i=start, we re-evaluate the now-merged block.
                    # Since it merged with previous, it is now part of a larger block.
                    # We need to find the start of that larger block to check its total duration?
                    # No, let's just continue.
                    i = end
        else:
            i += 1
            
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

def create_enhanced_prompt_v2(
    telemetry_summary: str,
    transcript_text: str,
    candidate_segments: List[Dict[str, Any]],
    plane_type: str = "Unknown",
    profile: dict = None
) -> str:
    """
    Improved prompt with better context and clearer instructions
    """
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
*Hints:*
- **GROUND_HIGH_RPM**: Likely the main Run-up mag check.
- **STATIONARY**: Could be Preflight, or the "Idle Check" phase of Run-up (if it follows High RPM).
- **TAXI**: Moving on ground. Likely Taxi Out or Taxi to Runway.

ALLOWED STATES (You MUST pick from this list ONLY):
- TAXI (TAXI TO RUNWAY, TAXI TO SHUTDOWN): Movement from on the ground, before takeoff and run up, after landing, and in between landings and takeoff. (Ground, < 25kts)
- RUNUP: Engine run-up, mag checks, cycling prop. (Ground, 0 speed, High RPM)
- TAKEOFF: Takeoff roll and initial climb to 500' AGL. (Ground -> Air, High RPM, Accel). IMPORTANT: Include ~10 seconds BEFORE liftoff (taxi onto runway, lineup) and ~10 seconds AFTER liftoff (initial climb). The segment should provide context around the takeoff event itself.
- SUSTAINED_CLIMB: Climb from 500' AGL to Cruise Altitude or Maneuver Altitude.
- CRUISE: Level flight for transit.
- MANEUVERS: General category for airwork (Steep Turns, Stalls, Slow Flight, etc.). *Prefer specific maneuver names if possible, e.g., STEEP_TURNS, SLOW_FLIGHT, POWER_OFF_STALL, POWER_ON_STALL.*
- SUSTAINED_DESCENT: Descent from altitude to traffic pattern altitude.
- TRAFFIC_PATTERN: Operations in the airport pattern (Downwind, Base, Final). Includes any all prep for landing procedures and actions. Try and see context of what airport may be being flown into and use knowledge of its traffic pattern altitude and procedures as context.
- LANDING: Final approach flare, touchdown, and roll-out. IMPORTANT: Include ~10 seconds BEFORE touchdown (short final, flare) and ~10 seconds AFTER touchdown (roll-out, turn off runway). The segment should capture the full landing context.
- SHUTDOWN: Engine shutdown and securing.

STATE TRANSITION LOGIC (Follow this flow):
1. TAXI -> RUNUP, TAKEOFF, SHUTDOWN
2. RUNUP -> TAXI, TAKEOFF
3. TAKEOFF -> SUSTAINED_CLIMB (after 500' AGL or so) or TRAFFIC_PATTERN
4. SUSTAINED_CLIMB -> CRUISE or MANEUVERS or TRAFFIC_PATTERN or SUSTAINED_DESCENT
5. CRUISE <-> MANEUVERS (Can switch back and forth)
6. CRUISE, MANEUVERS -> SUSTAINED_DESCENT, TRAFFIC_PATTERN, SUSTAINED_CLIMB
7. SUSTAINED_DESCENT -> TRAFFIC_PATTERN (or directly to LANDING) or CRUISE or MANUEVERS or SUSTAINED_CLIMB
8. TRAFFIC_PATTERN -> LANDING, SUSTAINED_DESCENT, SUSTAINED_CLIMB, CRUISE
9. LANDING -> TAXI_IN, TAKEOFF (if Touch & Go)

CRITICAL RULES:
1. **Granularity**: Do NOT lump all ground ops into "TAXI". You MUST distinguish between TAXI and RUNUP.
2. **Run-up Detection**: Look for High RPM (for the model of plane) with 0 Ground Speed. Audio sample cues: "mags", "checks", "runup", "RPM".
3. **Run-up Termination**: The RUNUP phase includes the high-RPM check AND the subsequent return to idle (idle check). It ends ONLY when:
   - The pilot announces "runup complete", "ready for takeoff", or calls tower.
   - OR the aircraft begins significant movement (Taxi to Runway).
   - DO NOT end the RUNUP segment just because RPM drops; wait for the "complete" call or movement. Going to idle signifies that the run up is ending soon.
4. **Takeoff vs Climb**: TAKEOFF ends when the aircraft is established in a climb (approx 500' AGL). Then switch to SUSTAINED_CLIMB.
5. **Pattern Work**: If the flight stays local and in the pattern, it might go TAKEOFF -> SUSTAINED_CLIMB -> TRAFFIC_PATTERN -> SUSTAINED_DESCENT -> LANDING -> TAKEOFF... and back again.
6. **Transcript is Key**: Use pilot calls ("Turning base", "Clear of runway", "Start taxi") to pinpoint transitions.

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


def identify_key_events_llm(transcript_text, telemetry_summary, plane_type, profile):
    """Ask LLM to identify ONLY major events: engine start, takeoff, landing, shutdown."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
        
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Identify ONLY these major flight events:
- Engine Start
- Takeoff (wheels leave ground)
- Landing (touchdown)
- Engine Shutdown

For each event, provide:
1. Event name
2. Approximate time (±10 seconds is fine)
3. Evidence from transcript and telemetry

Do NOT try to identify every segment. Just these 4 key events. Multiple takeoffs and landings may be present through the flight. Shutdown will happen only at the end after the final landing.

INPUT DATA:
Telemetry Summary:
{telemetry_summary}

Transcript:
{transcript_text}

OUTPUT JSON:
{{
  "events": [
    {{ "name": "TAKEOFF", "time": 120, "evidence": "Speed > 50kts, climbing" }},
    ...
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a flight data expert. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("events", [])
    except Exception as e:
        print(f"Key event detection error: {e}")
        return []

def fill_segments_between_events(key_events, candidates, total_duration):
    """
    Use heuristics (candidates) to fill in the gaps between key events.
    This is a simplified version - in reality we might want more complex logic.
    For now, we will just pass the key events to the next stage LLM as "Anchors".
    """
    # We can just return the candidates, but maybe annotate them if they overlap with key events?
    return candidates

def detect_segments_v2(
    transcript: dict, telemetry: dict, offset_sec: float = 0, plane_type: str = "Unknown", profile: dict = None
) -> List[Dict[str, Any]]:
    """
    Two-Stage LLM Approach for Segmentation.
    """
    print("SENSOR FUSION FLIGHT SEGMENTATION V2")
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

    # Stage 1: Identify Key Events
    print("Stage 1: Identifying Key Events...")
    key_events = identify_key_events_llm(transcript_text, telemetry_summary, plane_type, profile)
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
        
        prompt = create_enhanced_prompt_v2(
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

def detect_segments(
    transcript: dict, telemetry: dict, offset_sec: float = 0, plane_type: str = "Unknown", profile: dict = None
) -> List[Dict[str, Any]]:
    """
    Main entry point for Sensor Fusion segmentation.
    Delegates to V2.
    """
    return detect_segments_v2(transcript, telemetry, offset_sec, plane_type, profile)


def verify_llm_segments(
    llm_segments: List[Dict[str, Any]], telemetry: dict
) -> List[Dict[str, Any]]:
    """
    Verify the plausibility of LLM-generated segments against telemetry data.
    (Placeholder for now)
    """
    # TODO: Implement verification logic
    return llm_segments


# ============================================================================
# SEGMENT BUFFERING
# ============================================================================

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
    sorted_segs = sorted(segments, key=lambda x: x.get("start_time", 0))
    buffered = []
    
    for i, seg in enumerate(sorted_segs):
        new_seg = seg.copy()
        name = seg.get("name", "").upper()

        if name == "TAKEOFF":
            # Extend start backwards
            original_start = seg["start_time"]
            new_start = max(0, original_start - BUFFER_SECONDS)
            new_seg["start_time"] = new_start
            
            # Extend end forward
            original_end = seg["end_time"]
            max_end = total_duration if i + 1 >= len(sorted_segs) else sorted_segs[i + 1]["end_time"] - 5
            new_seg["end_time"] = min(original_end + BUFFER_SECONDS, max_end)
            
            print(f"Buffered TAKEOFF: {original_start}-{original_end}s -> {new_seg['start_time']}-{new_seg['end_time']}s")

        elif name == "LANDING":
            # Extend start backwards
            original_start = seg["start_time"]
            min_start = 0 if i == 0 else sorted_segs[i - 1]["start_time"] + 5
            new_seg["start_time"] = max(min_start, original_start - BUFFER_SECONDS)
            
            # Extend end forward
            original_end = seg["end_time"]
            max_end = total_duration if i + 1 >= len(sorted_segs) else sorted_segs[i + 1]["end_time"] - 5
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
        
    # 1. Sort
    sorted_segments = sorted(segments, key=lambda x: x["start_time"])
    
    # 2. Merge consecutive identical segments
    merged = []
    if sorted_segments:
        current = sorted_segments[0]
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

