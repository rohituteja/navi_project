# AI-Powered Flight Debrief Concept

An audio-first flight debrief web application designed to help pilots analyze their flight performance by combining rich telemetry data with cockpit audio.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **OpenAI API Key** (configured for `gpt-5-nano`)

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-your_api_key_here
```

### 2. Run Backend
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
*Backend runs on http://localhost:8000*

### 3. Run Frontend
```bash
cd frontend
npm install
npm run dev
```
*Frontend runs on http://localhost:5173*

---

## 🏗 Architecture & Design Decisions

The application is built on a **decoupled client-server architecture**, designed to handle the specific challenges of synchronizing multi-modal data (audio + time-series telemetry).

### High-Level Data Flow
1.  **Ingestion**: User uploads Flight Telemetry (**Garmin G3X Excel** or **G1000 CSV**) and Cockpit Audio (MP3/WAV) and selects the **Aircraft Profile**.
2.  **Normalization**: Backend standardizes data formats into a unified telemetry schema, handling differences between G3X and G1000 logs.
3.  **Transcription**: OpenAI Whisper converts audio to text with timestamps.
4.  **Robust Alignment**: The system synchronizes audio and telemetry using a **Multi-Detector Clustering** algorithm to find the optimal time offset.
5.  **Sensor Fusion Analysis**: The core engine combines physical data with semantic audio data to segment the flight using a two-stage AI process, guided by aircraft-specific performance data.
6.  **Visualization**: Frontend renders the aligned timeline and per-segment debriefs, allowing pilots to "replay" and analyze their flight.

### 🧠 The "Sensor Fusion" Engine (Backend)
The application uses a **Hybrid AI + Heuristic approach** for flight segmentation (`backend/app/services/segmentation.py`).

#### **Why this approach?**
-   **Pure Heuristics** are excellent for detecting physical states (e.g., "Airspeed > 70kts", "RPM > 5000") but lack semantic context (e.g., distinguishing a "Touch and Go" from a "Full Stop").
-   **Pure LLMs** excel at understanding context (e.g., Pilot says "Clearance delivery, Sling 123...") but struggle with precise mathematical boundaries in raw telemetry.

#### **How it works:**
1.  **Aircraft-Aware Heuristics**:
    -   A fast, physics-based pass identifies "Regions of Interest" (ROI) based on **Aircraft Profiles** (e.g., Sling NGT vs. Cessna 172S).
    -   Detects key physical events like engine start, rotation, and touchdown.
2.  **Two-Stage LLM Analysis**:
    -   **Stage 1 (Key Events)**: Identifies major anchor events (Takeoff, Landing) to establish a global timeline.
    -   **Stage 2 (Refinement)**: Feeds the heuristic candidates, telemetry summaries, and audio transcripts into **GPT-5-nano**.
    -   The LLM acts as a "Flight Instructor", using semantic cues ("Turning base") to refine the physical boundaries found by heuristics, enforcing a strict flight state machine.

### 🔗 Robust Audio-Telemetry Alignment (`alignment.py`)
Synchronizing separate cockpit audio with flight logs is difficult due to clock drift. We solved this with a **Multi-Pass Clustering Strategy**:
1.  **Detector Suite**: 9 specialized detectors look for correlation points (e.g., "Power/RPM Changes", "Airspeed Callouts", "Stall Warnings").
2.  **Clustering & Voting**: Candidates are clustered to find a consensus time offset. High-confidence events (like a distinct RPM spike during run-up) are weighted more heavily.

### 📊 Telemetry & Profiles
-   **Multi-Platform Support**: Native support for **Garmin G3X** and **Garmin G1000** data formats.
-   **Aircraft Profiles**: Customizable performance envelopes via `aircraft_profiles.json` (includes Sling NGT, Cessna 172S G1000, etc.).

---

## 🛠 Technology Stack

### Backend (Python + FastAPI)
-   **FastAPI**: For high-performance async API endpoints.
-   **Pandas**: For robust time-series telemetry manipulation and normalization.
-   **OpenAI GPT-5-nano**: Powers the core segmentation logic, event validation, and debrief generation.

### Frontend (React + Vite)
-   **React**: Component-based architecture for the flight dashboard.
-   **Vite**: Ultra-fast development and build tool.
-   **Vanilla CSS**: Custom "Aviation Dark Mode" aesthetic with smooth transitions and responsive layout.

## 📂 Project Structure

```
ai_flight_debrief_project/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes
│   │   ├── services/       # Core Logic (Segmentation, Alignment, Telemetry, Debrief)
│   │   └── config/         # Aircraft Profiles and settings
│   ├── main.py             # Entry point
│   └── requirements.txt
├── frontend/
│   ├── src/                # React components and logic
│   └── package.json
└── README.md
```

