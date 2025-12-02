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

def generate_candidates(telemetry: dict) -> List[Dict[str, Any]]:
    """
    Slices the flight into 'Regions of Interest' (ROI) based on physical state changes.
    """
    data_points = telemetry.get("data", [])
    if not data_points:
        return []

    candidates = []
    
    # We will iterate through the data and look for state changes
    # To keep it simple and robust, we'll classify each point and then merge consecutive points
    
    # 1. Point-wise classification
    point_classifications = []
    for i, p in enumerate(data_points):
        # Default to unknown
        label = "UNKNOWN"
        
        # Ground vs Air
        is_ground = p.get("is_ground", False)
        
        if is_ground:
            # Ground Logic
            if p.get("rpm", 0) > 1700:
                label = "GROUND_HIGH_RPM" # Potential Runup or Takeoff Roll
            else:
                label = "TAXI_OR_STATIONARY"
        else:
            # Air Logic
            roll_abs = abs(p.get("roll", 0))
            pitch_abs = abs(p.get("pitch", 0))
            v_spd = p.get("v_spd", 0)
            
            if roll_abs > 30:
                label = "MANEUVER_HIGH_BANK" # Steep turns
            elif pitch_abs > 15: # Arbitrary threshold for high pitch
                label = "MANEUVER_HIGH_PITCH"
            elif v_spd > 500:
                label = "CLIMB"
            elif v_spd < -500:
                label = "DESCENT"
            else:
                label = "STABLE_FLIGHT" # Cruise or gentle maneuvering
                
        point_classifications.append(label)
        
    # 2. Merge consecutive identical labels
    if not point_classifications:
        return []
        
    current_label = point_classifications[0]
    start_idx = 0
    
    for i in range(1, len(point_classifications)):
        label = point_classifications[i]
        if label != current_label:
            # End of segment
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
    
    # 3. Filter short noise (e.g. < 5 seconds) unless it's critical
    # Merging short segments into neighbors could be complex, for now just keeping them
    # but maybe filtering extremely short ones if they are just noise.
    # Let's keep them all for the LLM to decide, but maybe label them as "transient"
    
    return candidates

# ============================================================================
# ENHANCED LLM INTEGRATION
# ============================================================================

def create_enhanced_prompt_v2(
    telemetry_summary: str,
    transcript_text: str,
    candidate_segments: List[Dict[str, Any]],
    plane_type: str = "Unknown",
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

    prompt = f"""You are an expert flight instructor analyzing a flight training lesson. Your task is to identify and label specific flight segments for the entire flight, based on the provided telemetry and audio transcript data.

OBJECTIVE:
Create a chronological timeline of the flight phases using a STRICT STATE MACHINE approach.

AIRCRAFT CONTEXT:
Aircraft Type: {plane_type}
Use your knowledge of this specific aircraft's operating parameters (V-speeds, RPM ranges, performance characteristics).

INPUT DATA:

1. TELEMETRY SUMMARY (Sampled):
{telemetry_summary}

2. AUDIO TRANSCRIPT:
{transcript_text}

3. HEURISTIC CANDIDATES (Physics-based hints):
{candidate_text}

ALLOWED STATES (You MUST pick from this list ONLY):
- PREFLIGHT: Engine start, avionics setup, initial checks. (Ground, 0 speed)
- TAXI_OUT: Movement from ramp to run-up area. (Ground, < 25kts)
- RUNUP: Engine run-up, mag checks, cycling prop. (Ground, 0 speed, High RPM)
- TAXI_TO_RUNWAY: Movement from run-up area to runway threshold. (Ground, < 25kts)
- TAKEOFF: Takeoff roll and initial climb to 500' AGL. (Ground -> Air, High RPM, Accel)
- SUSTAINED_CLIMB: Climb from 500' AGL to Cruise Altitude or Maneuver Altitude.
- CRUISE: Level flight for transit.
- MANEUVERS: General category for airwork (Steep Turns, Stalls, Slow Flight, etc.). *Prefer specific maneuver names if possible, e.g., STEEP_TURNS, SLOW_FLIGHT, POWER_OFF_STALL, POWER_ON_STALL.*
- SUSTAINED_DESCENT: Descent from altitude to traffic pattern altitude.
- TRAFFIC_PATTERN: Operations in the airport pattern (Downwind, Base, Final). Includes any all prep for landing procedures and actions. Try and see context of what airport may be being flown into and use knowledge of its traffic pattern altitude and procedures as context.
- LANDING: Final approach flare, touchdown, and roll-out. You can include parts of the final approach right before the landing itself as well. 
- TAXI_IN: Taxi from runway to parking.
- SHUTDOWN: Engine shutdown and securing.

STATE TRANSITION LOGIC (Follow this flow):
1. PREFLIGHT -> TAXI_OUT
2. TAXI_OUT -> RUNUP, TAKEOFF
3. RUNUP -> TAXI_TO_RUNWAY, TAKEOFF
4. TAXI_TO_RUNWAY -> TAKEOFF
5. TAKEOFF -> SUSTAINED_CLIMB (after 500' AGL or so)
6. SUSTAINED_CLIMB -> CRUISE or MANEUVERS or TRAFFIC_PATTERN
7. CRUISE <-> MANEUVERS (Can switch back and forth)
8. CRUISE, MANEUVERS -> SUSTAINED_DESCENT, TRAFFIC_PATTERN
9. SUSTAINED_DESCENT -> TRAFFIC_PATTERN (or directly to LANDING) or CRUISE or MANUEVERS
10. TRAFFIC_PATTERN -> LANDING, SUSTAINED_DESCENT, SUSTAINED_CLIMB
11. LANDING -> TAXI_IN (or TAKEOFF if Touch & Go)
12. TAXI_IN -> SHUTDOWN, TAXI_OUT (to another TAKEOFF)

CRITICAL RULES:
1. **Granularity**: Do NOT lump all ground ops into "TAXI". You MUST distinguish between TAXI_OUT, RUNUP, and TAXI_TO_RUNWAY.
2. **Run-up Detection**: Look for High RPM (for the model of plane) with 0 Ground Speed. Audio sample cues: "mags", "checks", "runup", "RPM".
3. **Takeoff vs Climb**: TAKEOFF ends when the aircraft is established in a climb (approx 500' AGL). Then switch to SUSTAINED_CLIMB.
4. **Pattern Work**: If the flight stays local, it might go TAKEOFF -> TRAFFIC_PATTERN -> TOUCH_AND_GO -> TRAFFIC_PATTERN...
5. **Transcript is Key**: Use pilot calls ("Turning base", "Clear of runway", "Start taxi") to pinpoint transitions.

OUTPUT FORMAT:
Return a JSON object with a "segments" list.
{{
  "segments": [
    {{
      "name": "PREFLIGHT",
      "start_time": 0,
      "end_time": 45,
      "description": "Engine start and avionics setup",
      "confidence": 0.95
    }},
    ...
  ]
}}
"""
    return prompt


def detect_segments(
    transcript: dict, telemetry: dict, offset_sec: float = 0, plane_type: str = "Unknown"
) -> List[Dict[str, Any]]:
    """
    Main entry point for Sensor Fusion segmentation.
    """
    print("\n" + "=" * 60)
    print("SENSOR FUSION FLIGHT SEGMENTATION")
    print("=" * 60)
    print(f"✈️  Aircraft: {plane_type}")

    # 1. Data Ingestion & Pre-processing
    data_points = telemetry.get("data", [])
    if not data_points:
        return []

    total_duration = int(data_points[-1]["time_sec"])
    print(f"📊 Flight duration: {total_duration // 60}m {total_duration % 60}s")

    # 2. Heuristic Segmentation (Candidate Generation)
    print("🔍 Running Heuristic Candidate Generator...")
    candidates = generate_candidates(telemetry)
    print(f"   Found {len(candidates)} regions of interest.")

    # 3. Prepare Context for LLM
    # Telemetry Summary
    # Create a summarized version of telemetry for the prompt
    # We'll sample every 10 seconds or so, plus key events
    telemetry_summary_lines = []
    step = 10
    for i in range(0, len(data_points), step):
        p = data_points[i]
        # Format: T=100s Alt=2000 Spd=80 RPM=2300 Bank=10
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

    # 4. LLM Classification
    print(f"\n🤖 LLM analyzing flight with Sensor Fusion context...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No API key, returning heuristics only")
        return candidates # Fallback

    try:
        client = OpenAI(api_key=api_key)
        prompt = create_enhanced_prompt_v2(
            telemetry_summary, transcript_text, candidates, plane_type
        )

        response = client.chat.completions.create(
            model="gpt-5-nano",
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
        
        print(f"✓ LLM proposed {len(llm_segments)} segment(s)")

        # Verify LLM segments
        llm_segments = verify_llm_segments(llm_segments, telemetry)
        
        # Post-process: Merge and Fill Gaps
        final = post_process_segments(llm_segments, total_duration)
        
        print(f"\n✅ Total: {len(final)} segments")

        return final

    except Exception as e:
        print(f"❌ LLM error: {e}")
        return candidates


def verify_llm_segments(
    llm_segments: List[Dict[str, Any]], telemetry: dict
) -> List[Dict[str, Any]]:
    """
    Verify the plausibility of LLM-generated segments against telemetry data.
    (Placeholder for now)
    """
    # TODO: Implement verification logic
    return llm_segments


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

    return merged

