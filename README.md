# Navi Flight Debrief App

An audio-first flight approach debrief web application designed to help pilots analyze their approaches by combining flight telemetry with cockpit audio.

## Architecture & Design Decisions

### High-Level Overview
The application follows a decoupled client-server architecture. The backend handles heavy lifting (file processing, audio normalization, AI integration), while the frontend provides a responsive interface for uploading data and visualizing results.

We implemented an **asynchronous job pattern** for the analysis pipeline. Since audio transcription and flight data analysis can take time, the backend accepts the upload, returns a `job_id`, and processes the data in the background. The frontend polls for status updates, ensuring the UI remains responsive.

### Technology Stack

#### Backend: FastAPI (Python)
- **Why:** Python is the native language of data science and AI. FastAPI was chosen for its high performance (async support), automatic OpenAPI documentation, and ease of use with Pydantic for data validation.
- **Key Libraries:**
    - `pandas`: For robust parsing and manipulation of flight telemetry data (CSV/Excel).
    - `ffmpeg` (via `subprocess`): The industry standard for audio processing. We use it to normalize user audio to 16kHz mono MP3s to optimize for OpenAI's file size limits and API requirements.
    - `openai`: For accessing state-of-the-art transcription (Whisper) and reasoning models.

#### Frontend: React + Vite
- **Why:** React allows us to build a modular, component-based UI that can easily scale as we add more complex visualizations (charts, maps) for flight data. Vite provides a lightning-fast development experience.
- **Styling:** Vanilla CSS is used for maximum control and lightweight styling, focusing on a clean, dark-mode aesthetic suitable for aviation contexts.

### Key Features & Implementation Details

1.  **Audio Normalization**: Before transcription, all audio is normalized to ensure consistent quality and compliance with API limits.
2.  **Smart Caching**: To speed up development and save on API costs, we implemented a caching layer (`cached_transcript.json`) that stores the result of the last successful transcription.
3.  **LLM-Based Alignment**: Uses GPT-4o-mini to intelligently align audio transcripts with telemetry data by identifying key flight events (run-ups, takeoffs) and calculating accurate time offsets.
4.  **Aligned Data Visualization**: The primary interface displays transcript segments side-by-side with corresponding flight telemetry, making it easy to correlate what was said with what the aircraft was doing.

## Current Features

- **Multi-File Upload**: Seamlessly upload flight telemetry (CSV/XLSX) and cockpit audio (MP3/WAV/M4A) in a single interface.
- **Automated Audio Processing**: 
    - Automatically normalizes various audio formats to optimized 16kHz mono MP3s.
    - Handles large files efficiently using `ffmpeg`.
- **AI-Powered Transcription**: 
    - Integrates with OpenAI's Whisper model for high-accuracy speech-to-text.
    - Provides timestamped segments to correlate audio with flight events.
- **Intelligent Alignment**: 
    - Uses LLM-based analysis to automatically synchronize audio transcript with flight telemetry.
    - Identifies key events (run-up, takeoff) to calculate accurate time offsets.
- **Aligned Data Viewer**: 
    - **The hero feature** - displays transcript segments alongside corresponding telemetry data.
    - Shows real-time flight parameters (altitude, airspeed, vertical speed, heading, pitch, roll, flaps, RPM) for each moment in the audio.
    - Enables easy verification of audio-telemetry alignment and event correlation.
- **Flight Segmentation**: Automatically detects and segments different phases of flight (taxi, takeoff, cruise, landing, etc.).

## Structure

- `backend/`: FastAPI application for handling file uploads, transcription, and analysis.
- `frontend/`: React application (Vite) for the user interface.

## Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**: Required for the backend. [Download Python](https://www.python.org/downloads/)
- **Node.js 18+**: Required for the frontend. [Download Node.js](https://nodejs.org/)
- **FFmpeg**: Required for audio normalization.
    - **Mac**: `brew install ffmpeg`
    - **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.
    - **Linux**: `sudo apt install ffmpeg`

### 1. Environment Configuration

The application requires an OpenAI API key for transcription.

1.  Create a file named `.env` in the **root** directory of the project.
2.  Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your_api_key_here
```

### 2. Backend Setup

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```

2.  **Create a Virtual Environment** (Highly Recommended):
    It is best practice to use a virtual environment to manage dependencies and avoid conflicts.
    ```bash
    python3.11 -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    - **Mac/Linux**:
        ```bash
        source venv/bin/activate
        ```
    - **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Server**:
    ```bash
    uvicorn main:app --reload
    ```
    The backend API will be available at `http://localhost:8000`.

### 3. Frontend Setup

1.  Open a new terminal window and navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2.  **Install Dependencies**:
    ```bash
    npm install
    ```

3.  **Run the Development Server**:
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173` (or the URL shown in the terminal).
