import { AlertTriangle, CheckCircle, Info } from 'lucide-react';
import type { Forecast } from '../api';

interface Props {
  forecast: Forecast | null;
  loading: boolean;
  error: string;
  useTinyML: boolean;
  onToggleModel: (tiny: boolean) => void;
  basedOn: string;
}

const fmtValue = (v: number, unit: string) =>
  `${v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(v < 10 ? 2 : 1)}${unit ? ` ${unit}` : ''}`;

export default function ForecastPanel({ forecast, loading, error, useTinyML, onToggleModel, basedOn }: Props) {
  const raw = forecast?.raw;
  const drivers = raw?.explanation.slice(0, 6) ?? [];
  const scale = Math.max(5, ...drivers.map(d => Math.abs(d.contribution_pct)));
  const atRisk = raw?.risk_level === 'High';
  const severity = !raw ? 'good' : raw.risk_probability_pct >= 50 ? 'critical'
    : raw.risk_probability_pct >= 35 ? 'warning' : 'good';

  return (
    <section className="card forecast" aria-busy={loading}>
      <div className="card-head">
        <div>
          <h2>Tomorrow's activity forecast</h2>
          <p className="sub">Read from the last 3 days the tracker was worn{basedOn && ` (${basedOn})`}</p>
        </div>
        <div className="segmented" role="group" aria-label="Model">
          <button className={!useTinyML ? 'active' : ''} aria-pressed={!useTinyML}
            onClick={() => onToggleModel(false)}>CNN</button>
          <button className={useTinyML ? 'active' : ''} aria-pressed={useTinyML}
            onClick={() => onToggleModel(true)}>TinyML</button>
        </div>
      </div>

      {error && <div className="inline-error"><AlertTriangle size={16} /> {error}</div>}
      {!raw && !error && <div className="skeleton" style={{ height: 220 }} />}

      {raw && (
        <div className={`forecast-body ${loading ? 'stale' : ''}`}>
          <div className="forecast-hero">
            <div className="hero-label">Chance of a low-activity day</div>
            <div className="hero-value">{Math.round(raw.risk_probability_pct)}%</div>
            <div className={`meter ${severity}`} role="img"
              aria-label={`${raw.risk_probability_pct}% chance of a low-activity day`}>
              <div className="meter-fill" style={{ width: `${raw.risk_probability_pct}%` }} />
            </div>
            <div className={`status ${atRisk ? 'critical' : 'good'}`}>
              {atRisk ? <AlertTriangle size={15} /> : <CheckCircle size={15} />}
              {atRisk ? 'Low-activity day expected' : 'Active day expected'}
            </div>
            <p className="fine">
              <Info size={12} /> An activity forecast from {raw.model_type}, not a medical diagnosis.
              See below for how often it is right.
            </p>
          </div>

          <div className="drivers">
            <h3>What is moving the forecast</h3>
            <div className="driver-legend">
              <span><i className="swatch lowers" /> lowers risk</span>
              <span><i className="swatch raises" /> raises risk</span>
            </div>
            <ul>
              {drivers.map(d => {
                const width = (Math.abs(d.contribution_pct) / scale) * 50;
                const raises = d.contribution_pct > 0;
                return (
                  <li key={d.feature} className="driver"
                    title={`${d.label}: ${d.contribution_pct > 0 ? '+' : ''}${d.contribution_pct} points`}>
                    <div className="driver-name">
                      {d.label}
                      <span>
                        {d.placeholder
                          ? 'not recorded — placeholder used'
                          : `${fmtValue(d.patient_value, d.unit)} vs typical ${fmtValue(d.typical_value, d.unit)}`}
                      </span>
                    </div>
                    <div className="driver-track">
                      <div className={`driver-bar ${raises ? 'raises' : 'lowers'}`}
                        style={raises ? { left: '50%', width: `${width}%` } : { right: '50%', width: `${width}%` }} />
                    </div>
                    <div className="driver-value">
                      {d.contribution_pct > 0 ? '+' : ''}{d.contribution_pct.toFixed(1)} pts
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      )}

      {raw && raw.anomalies.length > 0 && (
        <div className="alerts">
          <h3>Out of range on the latest day worn</h3>
          <ul>
            {raw.anomalies.map(a => (
              <li key={a.field} className="status warning"><AlertTriangle size={14} /> {a.message}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
