import { useEffect, useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Activity, ShieldAlert, CheckCircle, Bot, Moon, Heart, Zap } from 'lucide-react';
import './App.css';

const API_BASE = "http://localhost:8000/api";

function App() {
  const [patients, setPatients] = useState<number[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<number | null>(null);
  const [healthData, setHealthData] = useState<any[]>([]);
  const [metricsChange, setMetricsChange] = useState<any>({});
  const [consequences, setConsequences] = useState<any[]>([]);
  const [chatLog, setChatLog] = useState<{ role: string, msg: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatRole, setChatRole] = useState('doctor');
  const [useTinyML, setUseTinyML] = useState(false);
  const [activeChart, setActiveChart] = useState<'activity' | 'sleep' | 'heart' | 'intensity'>('activity');

  useEffect(() => {
    axios.get(`${API_BASE}/patients`).then(res => {
      setPatients(res.data.patients);
      if (res.data.patients.length > 0) setSelectedPatient(res.data.patients[0]);
    });
  }, []);

  useEffect(() => {
    if (!selectedPatient) return;
    axios.get(`${API_BASE}/patients/${selectedPatient}/health_data`).then(res => {
      setHealthData(res.data.time_series);
      setMetricsChange(res.data.metrics_change);
      const recentSeq = res.data.time_series.slice(-3);
      axios.post(`${API_BASE}/predict_consequences`, {
        patient_id: selectedPatient,
        recent_sequence: recentSeq,
        use_tinyml: useTinyML
      }).then(r => setConsequences(r.data.predictions))
        .catch(e => console.error("Prediction error:", e));
    });
  }, [selectedPatient, useTinyML]);

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput;
    setChatLog(prev => [...prev, { role: 'user', msg }]);
    setChatInput('');
    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        message: msg, role: chatRole,
        patient_context: { patient_id: selectedPatient }
      });
      setChatLog(prev => [...prev, { role: 'bot', msg: res.data.reply }]);
    } catch (e) { console.error(e); }
  };

  const fmt = (val: number | undefined) =>
    val === undefined ? '-' : `${val > 0 ? '+' : ''}${val}%`;

  const chartTabs = [
    { key: 'activity', label: 'Activity', icon: <Activity size={14} /> },
    { key: 'sleep', label: 'Sleep', icon: <Moon size={14} /> },
    { key: 'heart', label: 'Heart Rate', icon: <Heart size={14} /> },
    { key: 'intensity', label: 'Intensity', icon: <Zap size={14} /> },
  ] as const;

  return (
    <div className="app-container">
      <aside className="chat-panel">
        <div className="brand"><Activity size={28} /><h2>HealthAI</h2></div>

        <div className="role-selector">
          <label>View As:</label>
          <select value={chatRole} onChange={e => setChatRole(e.target.value)}>
            <option value="doctor">Doctor</option>
            <option value="patient">Patient</option>
          </select>
        </div>

        <div className="model-toggle">
          <label>
            <input type="checkbox" checked={useTinyML}
              onChange={e => setUseTinyML(e.target.checked)} />
            Use TinyML (Edge)
          </label>
        </div>

        <div className="chat-window">
          {chatLog.length === 0 && (
            <div className="chat-hint">
              {chatRole === 'doctor'
                ? "Ask about steps, sleep, heart rate, BMI, sedentary time, risks, or overall summary."
                : "Ask how you're doing, about your sleep, steps, heart rate, or calories."}
            </div>
          )}
          {chatLog.map((log, i) => (
            <div key={i} className={`chat-bubble ${log.role}`}>{log.msg}</div>
          ))}
        </div>

        <div className="chat-input-area">
          <input type="text" placeholder="Ask health assistant..."
            value={chatInput} onChange={e => setChatInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendChat()} />
          <button onClick={sendChat}><Bot size={18} /></button>
        </div>
      </aside>

      <main className="dashboard">
        <header>
          <h1>Patient Health Dashboard</h1>
          <div className="patient-selector">
            <label>Patient: </label>
            <select value={selectedPatient || ''}
              onChange={e => setSelectedPatient(Number(e.target.value))}>
              {patients.map(p => <option key={p} value={p}>ID: {p}</option>)}
            </select>
          </div>
        </header>

        <section className="metrics-summary">
          {[
            { label: 'Steps', key: 'steps_change_pct' },
            { label: 'Calories', key: 'calories_change_pct' },
            { label: 'Distance', key: 'distance_change_pct' },
            { label: 'Sleep', key: 'sleep_change_pct' },
          ].map(({ label, key }) => (
            <div className="metric-card" key={key}>
              <h3>{label} Change</h3>
              <p className={(metricsChange[key] ?? 0) >= 0 ? 'positive' : 'negative'}>
                {fmt(metricsChange[key])}
              </p>
            </div>
          ))}
        </section>

        <section className="charts-area">
          <div className="chart-tabs">
            {chartTabs.map(tab => (
              <button key={tab.key}
                className={`chart-tab ${activeChart === tab.key ? 'active' : ''}`}
                onClick={() => setActiveChart(tab.key)}>
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          <div className="chart-container">
            {activeChart === 'activity' && (
              <>
                <h3>Activity Trends — Steps & Calories</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={healthData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="date" tick={{ fill: '#888' }} tickFormatter={v => v.slice(5)} />
                    <YAxis tick={{ fill: '#888' }} />
                    <Tooltip contentStyle={{ background: 'rgba(24, 24, 27, 0.8)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', backdropFilter: 'blur(12px)', color: '#fff' }} itemStyle={{ color: '#fff' }} cursor={{ fill: 'rgba(255, 255, 255, 0.05)', stroke: 'rgba(255, 255, 255, 0.1)' }} />
                    <Legend />
                    <Line type="monotone" dataKey="TotalSteps" stroke="#8b5cf6" name="Steps" dot={false} strokeWidth={2.5} activeDot={{ r: 6, fill: '#8b5cf6', stroke: '#fff' }} />
                    <Line type="monotone" dataKey="Calories" stroke="#10b981" name="Calories" dot={false} strokeWidth={2.5} activeDot={{ r: 6, fill: '#10b981', stroke: '#fff' }} />
                    <Line type="monotone" dataKey="TotalDistance" stroke="#3b82f6" name="Distance (km)" dot={false} strokeWidth={2} activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff' }} />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
            {activeChart === 'sleep' && (
              <>
                <h3>Sleep Trends — Minutes Asleep & Time in Bed</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={healthData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="date" tick={{ fill: '#888' }} tickFormatter={v => v.slice(5)} />
                    <YAxis tick={{ fill: '#888' }} />
                    <Tooltip contentStyle={{ background: 'rgba(24, 24, 27, 0.8)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', backdropFilter: 'blur(12px)', color: '#fff' }} itemStyle={{ color: '#fff' }} cursor={{ fill: 'rgba(255, 255, 255, 0.05)', stroke: 'rgba(255, 255, 255, 0.1)' }} />
                    <Legend />
                    <Bar dataKey="TotalMinutesAsleep" fill="#8b5cf6" name="Minutes Asleep" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="TotalTimeInBed" fill="#3b82f6" name="Time in Bed" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
            {activeChart === 'heart' && (
              <>
                <h3>Heart Rate — Avg / Max / Min</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={healthData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="date" tick={{ fill: '#888' }} tickFormatter={v => v.slice(5)} />
                    <YAxis tick={{ fill: '#888' }} domain={[40, 'auto']} />
                    <Tooltip contentStyle={{ background: 'rgba(24, 24, 27, 0.8)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', backdropFilter: 'blur(12px)', color: '#fff' }} itemStyle={{ color: '#fff' }} cursor={{ fill: 'rgba(255, 255, 255, 0.05)', stroke: 'rgba(255, 255, 255, 0.1)' }} />
                    <Legend />
                    <Line type="monotone" dataKey="AverageHeartrate" stroke="#ef4444" name="Avg HR" dot={false} strokeWidth={2.5} activeDot={{ r: 6, fill: '#ef4444' }} />
                    <Line type="monotone" dataKey="MaxHeartrate" stroke="#f87171" name="Max HR" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="MinHeartrate" stroke="#fca5a5" name="Min HR" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
            {activeChart === 'intensity' && (
              <>
                <h3>Activity Intensity — Active vs Sedentary Minutes</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={healthData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="date" tick={{ fill: '#888' }} tickFormatter={v => v.slice(5)} />
                    <YAxis tick={{ fill: '#888' }} />
                    <Tooltip contentStyle={{ background: 'rgba(24, 24, 27, 0.8)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', backdropFilter: 'blur(12px)', color: '#fff' }} itemStyle={{ color: '#fff' }} cursor={{ fill: 'rgba(255, 255, 255, 0.05)', stroke: 'rgba(255, 255, 255, 0.1)' }} />
                    <Legend />
                    <Bar dataKey="VeryActiveMinutes" fill="#10b981" name="Very Active" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="FairlyActiveMinutes" fill="#8b5cf6" name="Fairly Active" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="LightlyActiveMinutes" fill="#3b82f6" name="Lightly Active" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="SedentaryMinutes" fill="#ef4444" name="Sedentary" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </div>
        </section>

        <section className="consequences-area">
          <h3>
            AI Predictive Consequences
            {useTinyML && <span className="tinyml-badge">TinyML</span>}
          </h3>
          <div className="consequences-grid">
            {consequences.map((c, i) => (
              <div key={i} className={`consequence-card risk-${c.risk?.toLowerCase()}`}>
                <div className="card-header">
                  {c.risk === 'High' ? <ShieldAlert color="#ff4d4f" /> : <CheckCircle color="#52c41a" />}
                  <h4>{c.factor}</h4>
                </div>
                <p>{c.description}</p>
                <span className="risk-badge">{c.risk} Risk</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;