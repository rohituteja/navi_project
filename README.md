# AI-Powered Flight Debrief Concept

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
1.  **Ingestion**: User uploads Flight Telemetry (CSV/Excel) and Cockpit Audio (MP3/WAV) and selects the **Aircraft Profile**.
2.  **Normalization**: Backend standardizes data formats (standard telemetry schema).
3.  **Transcription**: OpenAI Whisper converts audio to text with timestamps.
4.  **Robust Alignment**: The system synchronizes audio and telemetry using a multi-detector clustering algorithm validated by an LLM.
5.  **Sensor Fusion Analysis**: The core engine combines physical data with semantic audio data to segment the flight using a two-stage AI process, guided by aircraft-specific performance data.
6.  **Visualization**: Frontend renders the aligned timeline, allowing pilots to "replay" the flight.

### 🧠 The "Sensor Fusion" Engine (Backend)
The most complex part of the system is the **Flight Segmentation Logic** (`backend/app/services/segmentation.py`). We chose a **Hybrid AI + Heuristic approach** rather than relying on just one method.

#### **Why this approach?**
-   **Pure Heuristics (Rule-based)** are good at detecting physical states (e.g., "High Engine RPMs", "Bank angle > 30°") but fail at understanding context (e.g., distinguishing a "Stop and Go" from a "Taxi back").
-   **Pure LLMs** are great at understanding context (e.g., Pilot says "Turning base") but struggle with precise timing and math.

#### **How it works:**
1.  **Aircraft-Aware Heuristics**:
    -   We first run a fast, physics-based pass over the telemetry using **Aircraft Profiles**.
    -   Thresholds for stalls, steep turns, and run-ups are dynamically adjusted based on the selected plane (e.g., Sling NGT vs. Cessna 172).
    -   *Result*: A list of "Regions of Interest" (ROI) with precise timestamps.
2.  **Two-Stage LLM Analysis**:
    -   **Stage 1 (Key Events)**: The LLM first identifies major anchor events (Engine Start, Takeoff, Landing, Shutdown) to establish a global timeline.
    -   **Stage 2 (Refinement)**: We feed the **Heuristic Candidates**, **Telemetry Summary**, **Audio Transcript**, and **Key Events** into the LLM.
    -   The LLM acts as a "Flight Instructor", using the semantic cues from the audio ("Clear of runway") to refine the physical boundaries found by the heuristics.
    -   It enforces a **Strict State Machine** (e.g., `PREFLIGHT` -> `TAXI` -> `RUNUP` -> `TAKEOFF`) to ensure logical flow between segments.

### 🔗 Robust Audio-Telemetry Alignment (`alignment.py`)
Synchronizing a separate audio recording with G3X flight logs is difficult due to clock drift and lack of common timestamps. We solved this with a **Multi-Pass Clustering Strategy**:

1.  **Candidate Detection**: We run 9 specialized detectors to find potential correlation points:
    -   `Power/RPM Changes` (e.g., "Full power" callout vs RPM spike)
    -   `Airspeed Callouts` (e.g., "Airspeed alive" vs Speed > 0)
    -   `Run-up Checks` (Distinctive high-RPM, zero-speed signature)
    -   `Takeoff Roll` & `Landings`
    -   `Steep Turns` & `Stall Warnings`
2.  **Clustering & Voting**: These candidates are clustered to find a consensus time offset. High-confidence events (like a distinct RPM spike during run-up) are weighted more heavily.
3.  **LLM Validation**: The proposed offset and top anchor points are sent to an LLM (GPT-5-mini) for a final sanity check to ensure physical and semantic consistency.

### 📊 Telemetry & Profiles
-   **Normalization**: We use `pandas` to create a unified internal schema from Garmin G3X Excel files.
-   **Aircraft Profiles**: The system loads `aircraft_profiles.json` to adapt its analysis logic. This allows it to correctly identify maneuvers based on the specific performance envelope of the plane being flown.

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
-   **OpenAI GPT-5-mini**: Used for high-speed, cost-effective semantic analysis and validation.

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
│   │   └── services/       # Core Logic (Segmentation, Alignment, Profiles)
│   ├── main.py             # App Entry Point
│   └── requirements.txt
├── frontend/
│   ├── src/                # React Components
│   └── package.json
└── README.md
```
