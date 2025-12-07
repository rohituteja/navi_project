import math
import statistics
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ============================================================================
# 1. HELPER FUNCTIONS & PREPROCESSING
# ============================================================================

def preprocess_transcript(transcript: dict) -> str:
    """Extracts segments into a concise text format: [Start-End] Text"""
    if not transcript or "segments" not in transcript:
        return ""
    lines = []
    for segment in transcript["segments"]:
        start = f"{segment['start']:.1f}"
        end = f"{segment['end']:.1f}"
        text = segment["text"].strip()
        lines.append(f"[{start}-{end}] {text}")
    return "\n".join(lines)

def get_telemetry_window(telemetry: dict, center_time: float, buffer_sec: float = 30.0) -> str:
    """Extracts a window of telemetry data around a center time."""
    if not telemetry or "data" not in telemetry:
        return ""
    start_time = center_time - buffer_sec
    end_time = center_time + buffer_sec
    lines = ["time,alt,speed,rpm,v_spd,flaps,roll"]
    for point in telemetry["data"]:
        t = point.get("time_sec", 0)
        if start_time <= t <= end_time:
            alt = f"{point.get('alt_agl', 0):.0f}"
            speed = f"{point.get('ias', 0):.0f}"
            rpm = f"{point.get('rpm', 0):.0f}"
            v_spd = f"{point.get('v_spd', 0):.0f}"
            flaps = f"{point.get('flaps', 0):.0f}"
            roll = f"{point.get('roll', 0):.0f}"
            lines.append(f"{t:.1f},{alt},{speed},{rpm},{v_spd},{flaps},{roll}")
    return "\n".join(lines) if len(lines) > 1 else "(No telemetry in this window)"

def find_transcript_matches(transcript: dict, keywords: List[str], window_start: float = 0, window_end: float = float('inf')) -> List[Dict]:
    """Finds all occurrences of keywords in the transcript."""
    matches = []
    if not transcript or "segments" not in transcript:
        return matches
        
    for seg in transcript["segments"]:
        if not (window_start <= seg["start"] <= window_end):
            continue
            
        text = seg["text"].lower()
        # Check for exact phrases or words
        matched_kws = [kw for kw in keywords if kw.lower() in text]
        if matched_kws:
            matches.append({
                "audio_time": seg["start"],
                "text": seg["text"],
                "matched_keywords": matched_kws
            })
    return matches

# ============================================================================
# 2. CANDIDATE DETECTORS
# ============================================================================

def find_power_rpm_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects power changes and specific RPM callouts."""
    candidates = []
    data = telemetry.get("data", [])
    if not data: return []

    # 1. Detect "Full Power" / High RPM
    # Telemetry: RPM > 4000 (or profile max)
    # Speech: "full power", "max power"
    high_rpm_threshold = 4000
    if profile and "rpm_profiles" in profile:
        high_rpm_threshold = profile["rpm_profiles"].get("run_up", [3800, 4200])[0]

    # Find telemetry events (rising edge)
    for i in range(10, len(data)-10):
        if data[i]["rpm"] > high_rpm_threshold and data[i-5]["rpm"] < high_rpm_threshold - 500:
            # Found power up
            matches = find_transcript_matches(transcript, ["full power", "max power", "power set", "airspeed alive"])
            for m in matches:
                candidates.append({
                    "type": "power_high",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.8,
                    "description": f"Full Power (RPM > {high_rpm_threshold}) matched '{m['text']}'"
                })

    # 2. Detect "Idle" / Low RPM
    # Telemetry: RPM drops < 2000
    # Speech: "idle", "power to idle"
    for i in range(10, len(data)-10):
        if data[i]["rpm"] < 2000 and data[i-5]["rpm"] > 2500:
            # Found power cut
            matches = find_transcript_matches(transcript, ["idle", "power to idle", "throttle closed", "pulling power"])
            for m in matches:
                candidates.append({
                    "type": "power_idle",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.85, # Idle calls are usually precise
                    "description": f"Power to Idle matched '{m['text']}'"
                })
                
    return candidates

def find_airspeed_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects airspeed callouts."""
    candidates = []
    data = telemetry.get("data", [])
    
    # 1. "Airspeed Alive" (approx 40kts or just moving)
    for i in range(10, len(data)-10):
        if data[i]["ias"] > 30 and data[i-5]["ias"] < 20:
            matches = find_transcript_matches(transcript, ["airspeed alive", "airspeed is alive"])
            for m in matches:
                candidates.append({
                    "type": "airspeed_alive",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.8,
                    "description": f"Airspeed Alive (>30kts) matched '{m['text']}'"
                })

    # 2. Specific speeds (60, 70, 80)
    target_speeds = {60: ["sixty", "60"], 70: ["seventy", "70"], 80: ["eighty", "80"]}
    for speed, kws in target_speeds.items():
        for i in range(10, len(data)-10):
            # Crossing speed threshold rising
            if data[i]["ias"] >= speed and data[i-1]["ias"] < speed:
                 matches = find_transcript_matches(transcript, kws + [f"{speed} knots"])
                 for m in matches:
                    candidates.append({
                        "type": f"speed_{speed}",
                        "telemetry_time": data[i]["time_sec"],
                        "audio_time": m["audio_time"],
                        "confidence": 0.6, # Can be ambiguous
                        "description": f"Speed {speed}kts matched '{m['text']}'"
                    })
    return candidates

def find_runup_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects run-up events."""
    candidates = []
    data = telemetry.get("data", [])
    
    runup_rpm_min = 3500
    if profile and "maneuver_thresholds" in profile:
        runup_rpm_min = profile["maneuver_thresholds"].get("rpm_runup_min", 3500)

    # Find run-up blocks in telemetry
    runup_times = []
    for p in data:
        if p["rpm"] > runup_rpm_min and p.get("gnd_spd", 0) < 5 and p["alt_agl"] < 50:
            runup_times.append(p["time_sec"])
            
    if not runup_times: return []
    
    # Group into events
    events = []
    if runup_times:
        start = runup_times[0]
        end = runup_times[0]
        for t in runup_times[1:]:
            if t - end < 5: end = t
            else:
                if end - start > 5: events.append((start, end))
                start = t
                end = t
        if end - start > 5: events.append((start, end))

    # Match with transcript
    keywords = ["run up", "run-up", "mag check", "magneto", "check mags"]
    matches = find_transcript_matches(transcript, keywords)
    
    for start, end in events:
        mid_time = (start + end) / 2
        for m in matches:
            # Run-up audio usually happens just before or during the RPM spike
            candidates.append({
                "type": "runup",
                "telemetry_time": start, # Start of RPM spike is best anchor
                "audio_time": m["audio_time"],
                "confidence": 0.9,
                "description": f"Run-up (RPM spike) matched '{m['text']}'"
            })
    return candidates

def find_takeoff_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects takeoff events."""
    candidates = []
    data = telemetry.get("data", [])
    
    # Detect takeoff roll start
    takeoff_times = []
    for i in range(10, len(data)-10):
        if data[i]["ias"] > 40 and data[i-5]["ias"] < 30 and data[i]["rpm"] > 4500:
             takeoff_times.append(data[i]["time_sec"])
             
    # Filter to first distinct takeoff (or multiple)
    distinct_takeoffs = []
    if takeoff_times:
        distinct_takeoffs.append(takeoff_times[0])
        for t in takeoff_times:
            if t - distinct_takeoffs[-1] > 60: distinct_takeoffs.append(t)
            
    keywords = ["takeoff", "taking off", "rolling", "full power", "clear for takeoff"]
    matches = find_transcript_matches(transcript, keywords)
    
    for t_time in distinct_takeoffs:
        for m in matches:
            candidates.append({
                "type": "takeoff",
                "telemetry_time": t_time,
                "audio_time": m["audio_time"],
                "confidence": 0.85,
                "description": f"Takeoff Roll matched '{m['text']}'"
            })
    return candidates

def find_climb_descent_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects climb and descent calls."""
    candidates = []
    data = telemetry.get("data", [])
    
    # Climb
    climb_kws = ["climb", "climbing", "positive rate"]
    climb_matches = find_transcript_matches(transcript, climb_kws)
    
    # Descent
    descent_kws = ["descend", "descending", "coming down"]
    descent_matches = find_transcript_matches(transcript, descent_kws)
    
    # We can't easily match every climb call to a specific second, 
    # but we can look for transitions to climb/descent.
    # For now, let's look for sustained VSPD changes.
    
    for i in range(10, len(data)-10):
        # Transition to Climb
        if data[i]["v_spd"] > 300 and data[i-5]["v_spd"] < 100:
            for m in climb_matches:
                candidates.append({
                    "type": "climb_start",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.5, # Vague
                    "description": f"Climb Start matched '{m['text']}'"
                })
        # Transition to Descent
        if data[i]["v_spd"] < -300 and data[i-5]["v_spd"] > -100:
             for m in descent_matches:
                candidates.append({
                    "type": "descent_start",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.5,
                    "description": f"Descent Start matched '{m['text']}'"
                })
    return candidates

def find_bank_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects steep turns."""
    candidates = []
    data = telemetry.get("data", [])
    
    for i in range(10, len(data)-10):
        if abs(data[i]["roll"]) > 40 and abs(data[i-5]["roll"]) < 20:
            matches = find_transcript_matches(transcript, ["steep turn", "45 degree", "bank", "turning left", "turning right"])
            for m in matches:
                candidates.append({
                    "type": "steep_turn",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.7,
                    "description": f"Steep Turn (>40deg) matched '{m['text']}'"
                })
    return candidates

def find_landing_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects landing pattern and touchdown."""
    candidates = []
    data = telemetry.get("data", [])
    
    # Touchdown detection (AGL < 10, neg VSPD -> 0 VSPD)
    touchdowns = []
    for i in range(10, len(data)-10):
        if data[i]["alt_agl"] < 10 and data[i-5]["alt_agl"] > 20:
            touchdowns.append(data[i]["time_sec"])
            
    # Filter distinct
    distinct_td = []
    if touchdowns:
        distinct_td.append(touchdowns[0])
        for t in touchdowns:
            if t - distinct_td[-1] > 60: distinct_td.append(t)
            
    # Touchdown keywords
    td_matches = find_transcript_matches(transcript, ["touchdown", "flare", "on the ground", "landing"])
    for t_time in distinct_td:
        for m in td_matches:
            candidates.append({
                "type": "touchdown",
                "telemetry_time": t_time,
                "audio_time": m["audio_time"],
                "confidence": 0.8,
                "description": f"Touchdown matched '{m['text']}'"
            })
            
    # Pattern calls (Downwind/Base/Final) - harder to map to exact telemetry without geo
    # But we can look for turns at pattern altitude (~1000ft AGL)
    # This is speculative, so low confidence
    return candidates

def find_taxi_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects taxi."""
    candidates = []
    data = telemetry.get("data", [])
    
    # Start of taxi (Speed > 0 on ground)
    for i in range(10, len(data)-10):
        if data[i].get("gnd_spd", 0) > 5 and data[i-5].get("gnd_spd", 0) < 1 and data[i]["is_ground"]:
            matches = find_transcript_matches(transcript, ["taxi", "taxiing", "brake check"])
            for m in matches:
                candidates.append({
                    "type": "taxi_start",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.7,
                    "description": f"Taxi Start matched '{m['text']}'"
                })
    return candidates

def find_stall_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Detects stalls."""
    candidates = []
    data = telemetry.get("data", [])
    
    # Stall: High Pitch, Low Speed, Drop in Alt/Pitch
    for i in range(10, len(data)-10):
        if data[i]["ias"] < 50 and data[i]["pitch"] < -5 and data[i-5]["pitch"] > 10:
            matches = find_transcript_matches(transcript, ["stall", "break", "buffet", "horn", "warning"])
            for m in matches:
                candidates.append({
                    "type": "stall_break",
                    "telemetry_time": data[i]["time_sec"],
                    "audio_time": m["audio_time"],
                    "confidence": 0.85,
                    "description": f"Stall Break matched '{m['text']}'"
                })
    return candidates

def generate_all_candidates(telemetry: dict, transcript: dict, profile: dict) -> List[Dict]:
    """Aggregates candidates from all detectors."""
    detectors = [
        find_power_rpm_candidates,
        find_airspeed_candidates,
        find_runup_candidates,
        find_takeoff_candidates,
        find_climb_descent_candidates,
        find_bank_candidates,
        find_landing_candidates,
        find_taxi_candidates,
        find_stall_candidates
    ]
    
    all_candidates = []
    for detector in detectors:
        try:
            cands = detector(telemetry, transcript, profile)
            all_candidates.extend(cands)
        except Exception as e:
            print(f"Detector {detector.__name__} failed: {e}")
            
    return all_candidates

# ============================================================================
# 3. OFFSET CALCULATION & VALIDATION
# ============================================================================

def cluster_and_vote_offsets(candidates: List[Dict], bin_size: float = 5.0) -> Tuple[float, float, List[Dict]]:
    """
    Groups offsets into bins and finds the consensus.
    Returns: (best_offset, confidence, used_candidates)
    """
    if not candidates:
        return 0.0, 0.0, []
        
    # Calculate implied offsets
    for c in candidates:
        c["implied_offset"] = c["telemetry_time"] - c["audio_time"]
        
    # Create bins
    bins = {}
    for c in candidates:
        offset = c["implied_offset"]
        bin_idx = int(offset / bin_size)
        if bin_idx not in bins:
            bins[bin_idx] = {"score": 0.0, "candidates": []}
        
        # Weight by confidence
        bins[bin_idx]["score"] += c["confidence"]
        bins[bin_idx]["candidates"].append(c)
        
    # Find winning bin
    if not bins:
        return 0.0, 0.0, []
        
    best_bin_idx = max(bins, key=lambda k: bins[k]["score"])
    best_bin = bins[best_bin_idx]
    
    # Calculate weighted average in winning bin
    total_score = best_bin["score"]
    weighted_sum = sum(c["implied_offset"] * c["confidence"] for c in best_bin["candidates"])
    consensus_offset = weighted_sum / total_score
    
    # Calculate confidence based on vote strength vs total candidates
    total_candidates_score = sum(c["confidence"] for c in candidates)
    confidence = min(0.95, total_score / total_candidates_score * 1.5) # Boost a bit if we have a clear cluster
    
    # Filter outliers from the winning bin candidates for the final list
    used_candidates = best_bin["candidates"]
    
    return consensus_offset, confidence, used_candidates

def validate_offset_with_physics(offset: float, telemetry: dict, transcript: dict) -> float:
    """
    Checks if the offset causes physics contradictions.
    Returns a confidence penalty factor (0.0 - 1.0).
    """
    # TODO: Implement detailed physics checks (e.g. "climbing" while VSPD < 0)
    # For now, just return 1.0 (no penalty)
    return 1.0

# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

def calculate_offset(transcript: dict, telemetry: dict, profile: dict = None) -> dict:
    """
    Main entry point for alignment.
    Uses multi-pass detection and clustering to determine the audio-telemetry offset.
    """
    print("STARTING ALIGNMENT")
    
    # 1. Generate Candidates
    candidates = generate_all_candidates(telemetry, transcript, profile)
    print(f"Found {len(candidates)} raw candidates.")
    
    # 2. Cluster and Vote
    consensus_offset, confidence, used_candidates = cluster_and_vote_offsets(candidates)
    print(f"Consensus Offset: {consensus_offset:.1f}s (Confidence: {confidence:.2f})")
    print(f"Based on {len(used_candidates)} agreeing candidates.")
    
    # Get top anchor points for reporting
    top_candidates = sorted(used_candidates, key=lambda x: x.get("confidence", 0), reverse=True)[:10]
    
    print(f"Final Offset: {consensus_offset:.1f}s")
    
    return {
        "offset_sec": consensus_offset,
        "confidence": confidence,
        "method": "multi_anchor_clustering",
        "reasoning": f"Determined from {len(used_candidates)} agreeing anchor points",
        "anchor_points": top_candidates
    }
