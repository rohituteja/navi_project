import React, { useState } from 'react';
import './index.css';

function FlightSegmentSection({ segment, transcriptSegments, alignmentOffset, findClosestTelemetryPoint, formatTimestamp }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="flight-segment">
      <div
        className="segment-header-collapsable"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="segment-title">
          <span className="segment-icon">{isExpanded ? '▼' : '▶'}</span>
          <h3>{segment.name.replace(/_/g, ' ')}</h3>
          <span className="segment-time">
            {formatTimestamp(segment.start_time)} - {formatTimestamp(segment.end_time)}
          </span>
        </div>
        <p className="segment-description">{segment.description}</p>
      </div>

      {isExpanded && (
        <div className="segment-content">
          {transcriptSegments.length > 0 ? (
            <div className="aligned-timeline">
              {transcriptSegments.map((seg, i) => {
                const flightTime = seg.start + alignmentOffset;
                const telemetryPoint = findClosestTelemetryPoint(flightTime);

                return (
                  <div key={i} className="aligned-segment">
                    <div className="segment-audio">
                      <div className="audio-timestamp">
                        Audio: {formatTimestamp(seg.start)} - {formatTimestamp(seg.end)}
                      </div>
                      <div className="audio-text">{seg.text}</div>
                    </div>

                    <div className="segment-telemetry">
                      <div className="flight-timestamp">
                        Flight Time: {formatTimestamp(flightTime)}
                      </div>
                      {telemetryPoint ? (
                        <div className="telemetry-grid">
                          <div className="telemetry-item">
                            <span className="label">Alt AGL:</span>
                            <span className="value">
                              {Math.round(telemetryPoint.alt_agl)}ft
                            </span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">IAS:</span>
                            <span className="value">{Math.round(telemetryPoint.ias)}kt</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">VS:</span>
                            <span className="value">{Math.round(telemetryPoint.v_spd)}fpm</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">HDG:</span>
                            <span className="value">{Math.round(telemetryPoint.heading)}°</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">Pitch:</span>
                            <span className="value">{telemetryPoint.pitch.toFixed(1)}°</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">Roll:</span>
                            <span className="value">{telemetryPoint.roll.toFixed(1)}°</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">Flaps:</span>
                            <span className="value">{Math.round(telemetryPoint.flaps)}°</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">RPM:</span>
                            <span className="value">{Math.round(telemetryPoint.rpm)}</span>
                          </div>
                          <div className="telemetry-item">
                            <span className="label">GS:</span>
                            <span className="value">
                              {Math.round(telemetryPoint.gnd_spd)}kt
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="telemetry-loading">Loading telemetry...</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="no-transcript">No transcript data for this segment</p>
          )}
        </div>
      )}
    </div>
  );
}


function App() {
  const [telemetryFile, setTelemetryFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [telemetryData, setTelemetryData] = useState(null);

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const findClosestTelemetryPoint = (flightTime) => {
    if (!telemetryData || !telemetryData.data) return null;

    // Find the data point closest to the given flight time
    let closest = null;
    let minDiff = Infinity;

    for (const point of telemetryData.data) {
      const diff = Math.abs(point.time_sec - flightTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
      // Since data is sorted, we can break early if difference starts increasing
      if (diff > minDiff) break;
    }

    return closest;
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
    setStatus('Processing your flight data...');

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

      // Automatically start analysis
      startAnalysis(id);
    } catch (err) {
      setError(err.message);
      setStatus('An error occurred while processing your flight data.');
    } finally {
      setLoading(false);
    }
  };

  const startAnalysis = async (id) => {
    setStatus('Analyzing flight...');

    const formData = new FormData();
    formData.append('job_id', id);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(`Analysis failed: ${errData.detail || response.statusText}`);
      }

      const data = await response.json();
      setAnalysis(data);
      console.log('Analysis received:', data);
      console.log('Segments:', data.segments);
      setStatus('Complete!');

      // Fetch full telemetry data
      fetchTelemetryData(id);
    } catch (err) {
      setError(err.message);
      setStatus('An error occurred during analysis.');
    }
  };

  const fetchTelemetryData = async (id) => {
    try {
      const response = await fetch(`http://localhost:8000/telemetry/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch telemetry data');
      }
      const data = await response.json();
      setTelemetryData(data);
      console.log('Telemetry data loaded:', data.data.length, 'points');
    } catch (err) {
      console.error('Error fetching telemetry:', err);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Navi Debrief Engine</h1>
        <p>Audio-first Flight Approach Debrief</p>
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
            {loading ? 'Processing...' : 'Analyze Flight'}
          </button>
          {error && <div className="error">{error}</div>}
          {status && <div className="status-message">{status}</div>}
        </section>

        {analysis && (
          <section className="results-section">
            <h2>Flight Debrief</h2>
            <div className="alignment-info">
              <p>
                <strong>Flight Duration:</strong>{' '}
                {formatTimestamp(analysis.telemetry.metadata.duration_sec || 0)}
                {' | '}
                <strong>Alignment:</strong> {analysis.alignment.method}
                (confidence: {(analysis.alignment.confidence * 100).toFixed(0)}%)
                {analysis.alignment.offset_sec !== 0 && (
                  <span> - Offset: {analysis.alignment.offset_sec.toFixed(1)}s</span>
                )}
              </p>
            </div>

            <p className="help-text">
              Click on each flight phase to expand and view the cockpit audio transcript with corresponding telemetry data.
            </p>

            <div className="flight-segments">
              {(analysis.segments && analysis.segments.length > 0 ? analysis.segments : [{
                name: "Full Flight",
                start_time: 0,
                end_time: analysis.telemetry.metadata.duration_sec || 0,
                description: "Full flight duration (no segments detected)."
              }])
                .map((segment, segIndex) => {
                  // Find all transcript segments that fall within this flight segment
                  const segmentTranscripts = transcript?.segments?.filter(transcriptSeg => {
                    const flightTime = transcriptSeg.start + analysis.alignment.offset_sec;
                    return flightTime >= segment.start_time && flightTime <= segment.end_time;
                  }) || [];

                  return { segment, segmentTranscripts, segIndex };
                })
                .filter(({ segmentTranscripts }) => segmentTranscripts.length > 0) // Only show segments with transcript data
                .map(({ segment, segmentTranscripts, segIndex }) => {
                  return (
                    <FlightSegmentSection
                      key={segIndex}
                      segment={segment}
                      transcriptSegments={segmentTranscripts}
                      alignmentOffset={analysis.alignment.offset_sec}
                      findClosestTelemetryPoint={findClosestTelemetryPoint}
                      formatTimestamp={formatTimestamp}
                    />
                  );
                })
              }
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
