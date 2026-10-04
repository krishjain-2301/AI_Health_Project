import { useCallback, useState } from 'react';
import type { ReactNode } from 'react';
import { Activity, AlertTriangle, Armchair, CheckCircle, Heart, Moon, RefreshCw } from 'lucide-react';
import { getChatStatus, getForecast, getHealthData, getModelInfo, getPatients } from './api';
import type { DayRow } from './api';
import { useFetch } from './useFetch';
import ChatPanel from './components/ChatPanel';
import ForecastPanel from './components/ForecastPanel';
import ModelPanel from './components/ModelPanel';
import TrendCharts from './components/TrendCharts';
import './App.css';

// Average over the days a metric was really recorded; null if it never was
function average(rows: DayRow[], key: keyof DayRow): number | null {
  const values = rows.map(r => r[key]).filter((v): v is number => typeof v === 'number');
  return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null;
}

interface Tile {
  label: string;
  icon: ReactNode;
  value: string | null;
  unit: string;
  ok: boolean;
  detail: string;
  change?: number;
  coverage?: string;
}

function StatTile({ t }: { t: Tile }) {
  if (t.value === null) {
    return (
      <div className="tile missing">
        <div className="tile-label">{t.icon}{t.label}</div>
        <div className="tile-value">—</div>
        <div className="tile-hint">Not recorded for this patient</div>
      </div>
    );
  }
  return (
    <div className="tile">
      <div className="tile-label">{t.icon}{t.label}</div>
      <div className="tile-value">{t.value}<span className="tile-unit">{t.unit}</span></div>
      <div className={`status ${t.ok ? 'good' : 'warning'}`}>
        {t.ok ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}{t.detail}
      </div>
      <div className="tile-hint">
        {t.change !== undefined && `${t.change > 0 ? '+' : ''}${t.change.toFixed(0)}% first to last day`}
        {t.change !== undefined && t.coverage && ' · '}
        {t.coverage}
      </div>
    </div>
  );
}

export default function App() {
  const [picked, setPicked]       = useState<number | null>(null);
  const [useTinyML, setUseTinyML] = useState(false);

  const patientsReq = useFetch(getPatients);
  const modelReq    = useFetch(getModelInfo);
  const chatReq     = useFetch(getChatStatus);
  const patients    = patientsReq.data;
  const selected    = picked ?? patients?.[0] ?? null;

  const fetchHealth   = useCallback(() => getHealthData(selected as number), [selected]);
  const fetchForecast = useCallback(() => getForecast(selected as number, useTinyML), [selected, useTinyML]);
  const healthReq     = useFetch(selected === null ? null : fetchHealth);
  const forecastReq   = useFetch(selected === null ? null : fetchForecast);

  const health      = healthReq.data;
  const healthError = healthReq.error;
  const bootError   = patientsReq.error;
  const retryBoot   = () => { patientsReq.reload(); modelReq.reload(); chatReq.reload(); };
  const retryData   = () => { healthReq.reload(); forecastReq.reload(); };

  if (bootError) {
    return (
      <div className="boot">
        <AlertTriangle size={28} />
        <h1>Dashboard can't load</h1>
        <p>{bootError}</p>
        <button className="btn" onClick={retryBoot}><RefreshCw size={15} /> Try again</button>
      </div>
    );
  }
  if (!patients) {
    return (
      <div className="boot">
        <div className="spinner" />
        <h1>Loading patient data</h1>
        <p>The first start reads the Fitbit files and trains the model, which can take a minute.</p>
      </div>
    );
  }

  const rows = health?.time_series ?? [];
  const quality = health?.data_quality;
  const steps = average(rows, 'TotalSteps');
  const sleep = average(rows, 'TotalMinutesAsleep');
  const hr    = average(rows, 'AverageHeartrate');
  const sed   = average(rows, 'SedentaryMinutes');
  const tiles: Tile[] = [
    { label: 'Average steps', icon: <Activity size={15} />, unit: 'per day',
      value: steps === null ? null : Math.round(steps).toLocaleString(),
      ok: (steps ?? 0) >= 7500, detail: (steps ?? 0) >= 7500 ? 'Meeting 7,500 goal' : 'Below 7,500 goal',
      change: health?.metrics_change.steps_change_pct },
    { label: 'Average sleep', icon: <Moon size={15} />, unit: 'hours a night',
      value: sleep === null ? null : (sleep / 60).toFixed(1),
      ok: (sleep ?? 0) >= 420, detail: (sleep ?? 0) >= 420 ? 'At least 7 hours' : 'Under 7 hours',
      coverage: quality && `${quality.sleep_days} of ${quality.total_days} nights logged` },
    { label: 'Average heart rate', icon: <Heart size={15} />, unit: 'bpm',
      value: hr === null ? null : Math.round(hr).toString(),
      ok: hr !== null && hr >= 60 && hr <= 100, detail: hr !== null && hr >= 60 && hr <= 100 ? 'Within 60–100 bpm' : 'Outside 60–100 bpm',
      coverage: quality && `${quality.heart_rate_days} of ${quality.total_days} days measured` },
    { label: 'Average sedentary time', icon: <Armchair size={15} />, unit: 'hours a day',
      value: sed === null ? null : (sed / 60).toFixed(1),
      ok: (sed ?? 0) <= 720, detail: (sed ?? 0) <= 720 ? 'Under 12 hours' : 'Over 12 hours',
      coverage: quality && quality.sleep_days === 0 ? 'includes sleep (none logged)' : undefined },
  ];
  const days = forecastReq.data?.raw.based_on ?? [];
  const basedOn = days.length ? `${days[0]} to ${days[days.length - 1]}` : '';

  return (
    <div className="app-container">
      <ChatPanel patientId={selected} status={chatReq.data} />

      <main className="dashboard">
        <header className="page-head">
          <div>
            <h1>Patient health dashboard</h1>
            <p className="sub">
              Fitbit tracker data{rows.length > 0 && `, ${rows[0].date} to ${rows[rows.length - 1].date}`}
              {quality && ` · worn ${quality.days_worn} of ${quality.total_days} days`}
            </p>
          </div>
          <label className="patient-selector">
            Patient
            <select value={selected ?? ''} onChange={e => setPicked(Number(e.target.value))}>
              {patients.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        </header>

        {healthError && (
          <div className="inline-error">
            <AlertTriangle size={16} /> {healthError}
            <button className="btn small" onClick={retryData}><RefreshCw size={13} /> Retry</button>
          </div>
        )}

        <ForecastPanel forecast={forecastReq.error ? null : forecastReq.data}
          loading={forecastReq.loading} error={forecastReq.error}
          useTinyML={useTinyML} onToggleModel={setUseTinyML} basedOn={basedOn} />

        {!health && !healthError && (
          <>
            <div className="tiles four">{[0, 1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 132 }} />)}</div>
            <div className="skeleton" style={{ height: 280 }} />
          </>
        )}

        {health && (
          // On a patient switch keep the previous patient's charts, dimmed, until the new data lands
          <div className={`patient-data ${healthReq.loading ? 'stale' : ''}`}>
            <section>
              <h2 className="section-title">Averages over the tracking period</h2>
              <div className="tiles four">{tiles.map(t => <StatTile key={t.label} t={t} />)}</div>
            </section>
            <section>
              <h2 className="section-title">Daily trends</h2>
              <TrendCharts data={rows} />
            </section>
          </div>
        )}

        {modelReq.data && <ModelPanel info={modelReq.data} useTinyML={useTinyML} />}
      </main>
    </div>
  );
}
