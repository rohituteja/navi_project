# Navi Flight Debrief App

An audio-first flight debrief web application designed to help pilots analyze their approaches by combining flight telemetry with cockpit audio.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**

- **OpenAI API Key**

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
1.  **Ingestion**: User uploads Flight Telemetry (CSV/Excel) and Cockpit Audio (MP3/WAV).
2.  **Normalization**: Backend standardizes data formats (standard telemetry schema).
3.  **Transcription**: OpenAI Whisper converts audio to text with timestamps.
4.  **Multi-Pass Alignment**: The system synchronizes audio and telemetry using multiple anchor points (Run-up, Takeoff, Landing) for high precision.
5.  **Sensor Fusion Analysis**: The core engine combines physical data with semantic audio data to segment the flight using a two-stage AI process.
6.  **Visualization**: Frontend renders the aligned timeline, allowing pilots to "replay" the flight.

### 🧠 The "Sensor Fusion" Engine (Backend)
The most complex part of the system is the **Flight Segmentation Logic** (`backend/app/services/segmentation.py`). We chose a **Hybrid AI + Heuristic approach** rather than relying on just one method.

#### **Why this approach?**
-   **Pure Heuristics (Rule-based)** are good at detecting physical states (e.g., "High Engine RPMs", "Bank angle > 30°") but fail at understanding context (e.g., distinguishing a "Stop and Go" from a "Taxi back").
-   **Pure LLMs** are great at understanding context (e.g., Pilot says "Turning base") but struggle with precise timing and math.

#### **How it works:**
1.  **Heuristic Candidate Generation**:
    -   We first run a fast, physics-based pass over the telemetry with temporal smoothing.
    -   *Simple Logic*: "If RPM > 2000 and GroundSpeed < 5kts with low altitude, it's likely a RUNUP. If AGL is less than 5 feet, it's likely we are on the ground, and probably in a TAXI state. High bank angles in the air mean we are either in a turn (during a CRUISE state) or a MANUEVER state, etc."
    -   *Result*: A list of "Regions of Interest" (ROI) with precise timestamps.
2.  **Two-Stage LLM Analysis**:
    -   **Stage 1 (Key Events)**: The LLM first identifies major anchor events (Engine Start, Takeoff, Landing, Shutdown) to establish a global timeline.
    -   **Stage 2 (Refinement)**: We feed the **Heuristic Candidates**, **Telemetry Summary**, **Audio Transcript**, and **Key Events** into the LLM.
    -   The LLM acts as a "Flight Instructor", using the semantic cues from the audio ("Clear of runway") to refine the physical boundaries found by the heuristics.
    -   It enforces a **Strict State Machine** (e.g., `PREFLIGHT` -> `TAXI` -> `RUNUP`) to ensure logical flow between segments.

### 📊 Telemetry Normalization (`telemetry.py`)
-   **Decision**: We use `pandas` to create a unified internal schema.
-   **Supported Format**: The system is optimized for Garmin G3X Excel files.
-   **Derived Metrics**: We calculate missing data like `turn_rate` or `is_ground` status on the fly to support the segmentation logic.

### ⚡ Asynchronous Job Pattern
Analyzing an hour-long flight takes time.
-   **Decision**: We avoid blocking HTTP requests.
-   **Flow**: `POST /upload` returns a `job_id` immediately. The frontend polls `GET /status/{job_id}`.
-   **Why**: Keeps the UI responsive and allows for progress updates (e.g., "Transcribing...", "Analyzing...").

---

## 🛠 Technology Stack

### Backend (Python + FastAPI)
-   **FastAPI**: Chosen for native async support (crucial for long-running AI tasks) and auto-generated OpenAPI docs.
-   **Pandas**: The industry standard for time-series data manipulation.


### Frontend (React + Vite)
-   **React**: Component-based architecture suitable for complex dashboards.
-   **Vite**: Extremely fast build tool.
-   **Vanilla CSS**: We intentionally avoided heavy UI frameworks to maintain full control over the "Dark Mode" aviation aesthetic.

## 📂 Project Structure

```
navi_project/
├── backend/
│   ├── app/
│   │   ├── api/            # API Endpoints
│   │   ├── services/       # Core Logic (Segmentation, Telemetry)
│   │   └── main.py         # App Entry Point
│   └── requirements.txt
├── frontend/
│   ├── src/                # React Components
│   └── package.json
└── README.md
```
