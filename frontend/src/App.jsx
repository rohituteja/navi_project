import React, { useState, useEffect } from 'react';
import './index.css';

function FlightSegmentSection({
  segment,
  segmentIndex,
  transcriptSegments,
  alignmentOffset,
  findClosestTelemetryPoint,
  formatTimestamp,
  jobId,
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [segmentDebrief, setSegmentDebrief] = useState(null);
  const [debriefLoading, setDebriefLoading] = useState(false);

  const handleAnalyzeSegment = async () => {
    setDebriefLoading(true);
    const formData = new FormData();
    formData.append('job_id', jobId);
    formData.append('segment_index', segmentIndex);

    try {
      const response = await fetch('http://localhost:8000/debrief/segment', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setSegmentDebrief(data.debrief);
      }
    } catch (err) {
      console.error('Error fetching segment debrief:', err);
    } finally {
      setDebriefLoading(false);
    }
  };

  return (
    <div className="flight-segment">
      <div className="segment-header-collapsable" onClick={() => setIsExpanded(!isExpanded)}>
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
          <button
            className="segment-analyze-btn"
            onClick={handleAnalyzeSegment}
            disabled={debriefLoading}
          >
            {debriefLoading ? 'Analyzing...' : 'Analyze This Segment'}
          </button>

          {debriefLoading && (
            <div className="segment-debrief-container">
              <div className="segment-debrief-loading">
                <span>Generating focused debrief...</span>
                <div className="spinner"></div>
              </div>
            </div>
          )}

          {segmentDebrief && (
            <div className="segment-debrief-container">
              <div className="segment-debrief-header">CFI Analysis</div>
              <p className="segment-debrief-text">{segmentDebrief}</p>
            </div>
          )}

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
                            <span className="value">{Math.round(telemetryPoint.alt_agl)}ft</span>
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
                            <span className="value">{Math.round(telemetryPoint.gnd_spd)}kt</span>
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
  const [planeType, setPlaneType] = useState('Sling Next Generation Trainer (NGT)');
  const [debrief, setDebrief] = useState(null);
  const [debriefLoading, setDebriefLoading] = useState(false);

  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    let interval;
    if (loading) {
      setElapsedTime(0);
      interval = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading]);

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
    if (!telemetryFile || !audioFile) {
      setError('Both telemetry and audio files are required.');
      return;
    }

    setLoading(true);
    setError(null);
    setStatus('Processing your flight data...');

    const formData = new FormData();
    formData.append('telemetry_file', telemetryFile);
    formData.append('plane_type', planeType);
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

      // Fetch Debrief
      fetchDebrief(id);
    } catch (err) {
      setError(err.message);
      setStatus('An error occurred during analysis.');
      setLoading(false);
    } finally {
      setLoading(false);
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

  const fetchDebrief = async (id) => {
    setDebriefLoading(true);
    const formData = new FormData();
    formData.append('job_id', id);

    try {
      const response = await fetch('http://localhost:8000/debrief', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setDebrief(data.debrief);
      }
    } catch (err) {
      console.error('Error fetching debrief:', err);
    } finally {
      setDebriefLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>AI-Powered Flight Debrief Concept</h1>
      </header>

      <main>
        <section className="upload-section">
          <div
            className="upload-header"
            onClick={() => analysis && setAnalysis(null)}
            style={{ cursor: analysis ? 'pointer' : 'default' }}
          >
            <h2>Upload Flight Data</h2>
            {analysis && (
              <div className="collapse-indicator-container">
                <span className="collapse-indicator">Click to upload new flight</span>
              </div>
            )}
          </div>
          {!analysis && (
            <>
              <div className="form-group">
                <label>Telemetry (Garmin G3X or G1000 .xlsx/.csv) *</label>
                <input
                  type="file"
                  accept=".xlsx,.csv"
                  onChange={(e) => handleFileChange(e, 'telemetry')}
                />
              </div>
              <div className="form-group">
                <label>Cockpit Audio (.mp3) *</label>
                <input type="file" accept=".mp3" onChange={(e) => handleFileChange(e, 'audio')} />
              </div>
              <div className="form-group">
                <label>Plane Type</label>
                <select
                  value={planeType}
                  onChange={(e) => setPlaneType(e.target.value)}
                  className="plane-select"
                >
                  <option value="Sling Next Generation Trainer (NGT)">
                    Sling Next Generation Trainer (NGT)
                  </option>
                  <option value="Cessna 172S (G1000)">
                    Cessna 172S (G1000)
                  </option>
                </select>
              </div>
              <button
                onClick={handleUpload}
                disabled={loading || !telemetryFile || !audioFile}
                className={`analyze-btn ${loading ? 'loading' : ''}`}
              >
                {loading ? (
                  <div className="btn-content">
                    <div className="spinner-small"></div>
                    <span className="btn-text">{status}</span>
                    <span className="btn-timer">{formatTimestamp(elapsedTime)}</span>
                  </div>
                ) : (
                  'Analyze Flight'
                )}
              </button>
              {error && <div className="error">{error}</div>}
            </>
          )}
        </section>

        {analysis && (
          <section className="results-section">
            <h2>Flight Debrief</h2>
            <div className="debrief-card">
              {debriefLoading ? (
                <div className="debrief-loading">
                  <p>Generating CFI Debrief...</p>
                  <div className="spinner"></div>
                </div>
              ) : debrief ? (
                <div className="debrief-content">
                  <h3>CFI Overview</h3>
                  <p>{debrief}</p>
                  <div className="debrief-meta">
                    <span>
                      Duration: {formatTimestamp(analysis.telemetry.metadata.duration_sec || 0)}
                    </span>
                    <span> • </span>
                    <span>Audio-Telemetry Offset: {analysis.alignment.offset_sec.toFixed(1)}s</span>
                  </div>
                </div>
              ) : (
                <div className="alignment-info">
                  <p>
                    <strong>Flight Duration:</strong>{' '}
                    {formatTimestamp(analysis.telemetry.metadata.duration_sec || 0)}
                    {' | '}
                    <strong>Audio-Telemetry Offset:</strong>{' '}
                    {analysis.alignment.offset_sec.toFixed(1)}s
                  </p>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                    Generating flight overview...
                  </p>
                </div>
              )}
            </div>

            <p className="help-text">
              Click on each flight phase to expand and view the cockpit audio transcript with
              corresponding telemetry data.
            </p>

            <div className="flight-segments">
              {(analysis.segments && analysis.segments.length > 0
                ? analysis.segments
                : [
                  {
                    name: 'Full Flight',
                    start_time: 0,
                    end_time: analysis.telemetry.metadata.duration_sec || 0,
                    description: 'Full flight duration (no segments detected).',
                  },
                ]
              )
                .map((segment, segIndex) => {
                  // Find all transcript segments that fall within this flight segment
                  const segmentTranscripts =
                    transcript?.segments?.filter((transcriptSeg) => {
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
                      segmentIndex={segIndex}
                      transcriptSegments={segmentTranscripts}
                      alignmentOffset={analysis.alignment.offset_sec}
                      findClosestTelemetryPoint={findClosestTelemetryPoint}
                      formatTimestamp={formatTimestamp}
                      jobId={jobId}
                    />
                  );
                })}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
