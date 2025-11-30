import React, { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [telemetryFile, setTelemetryFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cacheStatus, setCacheStatus] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Check cache status on mount
  useEffect(() => {
    checkCacheStatus();
  }, []);

  const checkCacheStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/cache/status');
      if (response.ok) {
        const data = await response.json();
        setCacheStatus(data);
      }
    } catch (err) {
      console.log('Could not check cache status:', err);
    }
  };

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleFileChange = (e, type) => {
    const file = e.target.files[0];
    if (type === 'telemetry') setTelemetryFile(file);
    if (type === 'audio') setAudioFile(file);
  };

  const handleUpload = async () => {
    if (!telemetryFile) {
      setError('Telemetry file is required.');
      return;
    }

    setLoading(true);
    setError(null);
    setStatus('Uploading...');

    const formData = new FormData();
    formData.append('telemetry_file', telemetryFile);
    if (audioFile) {
      formData.append('audio_file', audioFile);
    }

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setJobId(data.job_id);
      setStatus('queued');

      // Automatically start transcription
      startTranscription(data.job_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const startTranscription = async (id) => {
    setStatus('Transcribing...');
    const formData = new FormData();
    formData.append('job_id', id);

    try {
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(`Transcription failed: ${errData.detail || response.statusText}`);
      }

      const data = await response.json();
      setTranscript(data);

      // Get job status to check cache usage
      const statusResponse = await fetch(`http://localhost:8000/job/${id}/status`);
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        if (statusData.message) {
          setStatus(`done - ${statusData.message}`);
        } else {
          setStatus('done');
        }
      } else {
        setStatus('done');
      }

      // Refresh cache status
      checkCacheStatus();
    } catch (err) {
      setError(err.message);
      setStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const filteredSegments =
    transcript?.segments?.filter((seg) =>
      seg.text.toLowerCase().includes(searchQuery.toLowerCase())
    ) || [];

  return (
    <div className="container">
      <header>
        <h1>Navi Debrief Engine</h1>
        <p>Audio-first Flight Approach Debrief</p>
        {cacheStatus?.cached && <div className="cache-badge">✓ Cached transcript available</div>}
      </header>

      <main>
        <section className="upload-section">
          <h2>1. Upload Flight Data</h2>
          <div className="form-group">
            <label>Telemetry (.xlsx, .csv) *</label>
            <input
              type="file"
              accept=".xlsx,.csv"
              onChange={(e) => handleFileChange(e, 'telemetry')}
            />
          </div>
          <div className="form-group">
            <label>Cockpit Audio (.mp3, .wav) (Optional)</label>
            <input
              type="file"
              accept=".mp3,.wav,.m4a"
              onChange={(e) => handleFileChange(e, 'audio')}
            />
          </div>
          <button onClick={handleUpload} disabled={loading || !telemetryFile}>
            {loading ? 'Processing...' : 'Upload & Process'}
          </button>
          {error && <div className="error">{error}</div>}
        </section>

        {status && (
          <section className="status-section">
            <h2>
              Status: <span className={`status-${status}`}>{status}</span>
            </h2>
            {jobId && <p>Job ID: {jobId}</p>}
          </section>
        )}

        {transcript && (
          <section className="results-section">
            <h2>Transcript</h2>

            {transcript.segments && transcript.segments.length > 0 && (
              <div className="transcript-viewer">
                <div className="transcript-controls">
                  <input
                    type="text"
                    placeholder="Search transcript..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="search-input"
                  />
                  <span className="segment-count">
                    {filteredSegments.length} of {transcript.segments.length} segments
                  </span>
                </div>

                <div className="timeline">
                  {filteredSegments.map((seg, i) => (
                    <div key={i} className="timeline-segment">
                      <div className="timestamp">
                        {formatTimestamp(seg.start)} - {formatTimestamp(seg.end)}
                      </div>
                      <div className="segment-text">{seg.text}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
