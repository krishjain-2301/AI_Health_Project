import type { ReactElement } from 'react';
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { DayRow } from '../api';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtDate = (iso: string) => `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${Number(iso.slice(8, 10))}`;
const fmtInt = (v: number) => Math.round(v).toLocaleString();
const fmtHours = (min: number) => `${Math.floor(min / 60)}h ${Math.round(min % 60)}m`;

// Round axis ticks (1 / 2 / 5 x 10^n steps) covering 0..max in at most five steps
function niceTicks(max: number, base = 1): number[] {
  const rough = Math.max(max, base) / 5 / base;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = ([1, 2, 5, 10].find(m => m * pow >= rough) ?? 10) * pow * base;
  const ticks = [];
  for (let t = 0; t < max + step; t += step) ticks.push(t);
  return ticks;
}
const maxOf = (data: DayRow[], value: (d: DayRow) => number) => Math.max(0, ...data.map(value));

const axis = { tick: { fill: 'var(--text-muted)', fontSize: 11 }, tickLine: false, stroke: 'var(--axis)' };
const HEIGHT = 200;

interface TipProps {
  active?: boolean;
  label?: string;
  payload?: { dataKey: string; name: string; color: string; value: number | number[] }[];
  format: (v: number) => string;
}

function ChartTooltip({ active, payload, label, format }: TipProps) {
  if (!active || !payload?.length || !label) return null;
  return (
    <div className="tip">
      <div className="tip-date">{fmtDate(label)}</div>
      {payload.map(p => (
        <div className="tip-row" key={p.dataKey}>
          <i className="swatch" style={{ background: p.color }} />
          <span>{p.name}</span>
          <b>{Array.isArray(p.value) ? p.value.map(fmtInt).join('–') : format(p.value)}</b>
        </div>
      ))}
    </div>
  );
}

// Days outside the normal range get a marker; in-range days stay a plain line
interface DotProps { cx?: number; cy?: number; index?: number; payload?: DayRow }

const flaggedDot = (key: keyof DayRow, isFlagged: (v: number) => boolean) => (props: DotProps) => {
  const v = props.payload?.[key];
  if (typeof v !== 'number' || v === 0 || !isFlagged(v)) return <g key={props.index} />;
  return <circle key={props.index} cx={props.cx} cy={props.cy} r={4.5}
    fill="var(--status-critical)" stroke="var(--surface-1)" strokeWidth={2} />;
};

interface CardProps {
  title: string;
  note?: string;
  legend?: { label: string; color: string }[];
  empty?: string;
  wide?: boolean;
  children: ReactElement;
}

function ChartCard({ title, note, legend, empty, wide, children }: CardProps) {
  return (
    <div className={`card chart-card ${wide ? 'wide' : ''}`}>
      <div className="chart-head">
        <h3>{title}</h3>
        {note && !empty && <span className="chart-note">{note}</span>}
      </div>
      {legend && !empty && (
        <div className="legend">
          {legend.map(l => <span key={l.label}><i className="swatch" style={{ background: l.color }} />{l.label}</span>)}
        </div>
      )}
      {empty
        ? <div className="chart-empty" style={{ height: HEIGHT }}>{empty}</div>
        : <ResponsiveContainer width="100%" height={HEIGHT}>{children}</ResponsiveContainer>}
    </div>
  );
}

const grid = <CartesianGrid stroke="var(--grid)" vertical={false} />;
const xAxis = <XAxis dataKey="date" tickFormatter={fmtDate} minTickGap={28} {...axis} />;
const margin = { top: 12, right: 12, bottom: 0, left: 0 };
const refLabel = (value: string) => ({ value, position: 'insideTopLeft' as const, fill: 'var(--text-muted)', fontSize: 10 });

export default function TrendCharts({ data }: { data: DayRow[] }) {
  const hasSleep = data.some(d => d.TotalMinutesAsleep != null);
  const hasHR    = data.some(d => d.AverageHeartrate != null);
  const stepTicks   = niceTicks(maxOf(data, d => d.TotalSteps));
  const calTicks    = niceTicks(maxOf(data, d => d.Calories));
  const sleepTicks  = niceTicks(maxOf(data, d => d.TotalMinutesAsleep ?? 0), 60);
  const activeTicks = niceTicks(maxOf(data, d => d.VeryActiveMinutes + d.FairlyActiveMinutes + d.LightlyActiveMinutes));
  const yDomain = (ticks: number[]) => ({ ticks, domain: [0, ticks[ticks.length - 1]] });
  const hrData   = data.map(d => ({
    ...d, hrRange: d.MinHeartrate != null && d.MaxHeartrate != null ? [d.MinHeartrate, d.MaxHeartrate] : null,
  }));

  return (
    <div className="chart-grid">
      <ChartCard title="Daily steps" note="● under 3,000 or over 25,000">
        <LineChart data={data} margin={margin}>
          {grid}{xAxis}
          <YAxis {...axis} {...yDomain(stepTicks)} axisLine={false} width={44} tickFormatter={v => (v >= 1000 ? `${v / 1000}k` : v)} />
          <Tooltip content={<ChartTooltip format={fmtInt} />} cursor={{ stroke: 'var(--axis)' }} />
          <ReferenceLine y={7500} stroke="var(--text-muted)" strokeDasharray="4 3" label={refLabel('7,500 goal')} />
          <Line type="monotone" dataKey="TotalSteps" name="Steps" stroke="var(--series-1)" strokeWidth={2}
            dot={flaggedDot('TotalSteps', v => v < 3000 || v > 25000)} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ChartCard>

      <ChartCard title="Calories burned" note="kcal per day">
        <LineChart data={data} margin={margin}>
          {grid}{xAxis}
          <YAxis {...axis} {...yDomain(calTicks)} axisLine={false} width={44} tickFormatter={fmtInt} />
          <Tooltip content={<ChartTooltip format={(v: number) => `${fmtInt(v)} kcal`} />} cursor={{ stroke: 'var(--axis)' }} />
          <Line type="monotone" dataKey="Calories" name="Calories" stroke="var(--series-1)" strokeWidth={2}
            dot={false} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ChartCard>

      <ChartCard title="Sleep" note="time asleep per logged night · dashed line is 7 hours"
        empty={hasSleep ? undefined : 'No sleep was logged for this patient.'}>
        <BarChart data={data} margin={margin}>
          {grid}{xAxis}
          <YAxis {...axis} {...yDomain(sleepTicks)} axisLine={false} width={44}
            tickFormatter={v => `${v / 60}h`} />
          <Tooltip content={<ChartTooltip format={fmtHours} />} cursor={{ fill: 'var(--hover)' }} />
          <ReferenceLine y={420} stroke="var(--text-muted)" strokeDasharray="4 3" />
          <Bar dataKey="TotalMinutesAsleep" name="Asleep" fill="var(--series-1)" radius={[4, 4, 0, 0]}
            maxBarSize={24} isAnimationActive={false} />
        </BarChart>
      </ChartCard>

      <ChartCard title="Heart rate" note="daily average, band shows min–max · ● outside 60–100 bpm"
        empty={hasHR ? undefined : 'No heart-rate data was recorded for this patient.'}>
        <ComposedChart data={hrData} margin={margin}>
          {grid}{xAxis}
          <YAxis {...axis} axisLine={false} width={44} domain={[40, 'auto']} />
          <Tooltip content={<ChartTooltip format={(v: number) => `${Math.round(v)} bpm`} />} cursor={{ stroke: 'var(--axis)' }} />
          <Area type="monotone" dataKey="hrRange" name="Min–max" stroke="none" fill="var(--series-1)"
            fillOpacity={0.12} isAnimationActive={false} />
          <Line type="monotone" dataKey="AverageHeartrate" name="Average" stroke="var(--series-1)" strokeWidth={2}
            dot={flaggedDot('AverageHeartrate', v => v < 60 || v > 100)} activeDot={{ r: 5 }} isAnimationActive={false} />
        </ComposedChart>
      </ChartCard>

      <ChartCard wide title="Active minutes" note="per day, by intensity"
        legend={[
          { label: 'Very active', color: 'var(--series-1)' },
          { label: 'Fairly active', color: 'var(--series-2)' },
          { label: 'Lightly active', color: 'var(--series-3)' },
        ]}>
        <BarChart data={data} margin={margin}>
          {grid}{xAxis}
          <YAxis {...axis} {...yDomain(activeTicks)} axisLine={false} width={44} />
          <Tooltip content={<ChartTooltip format={(v: number) => `${fmtInt(v)} min`} />} cursor={{ fill: 'var(--hover)' }} />
          <Bar dataKey="VeryActiveMinutes" name="Very active" stackId="a" fill="var(--series-1)"
            stroke="var(--surface-1)" strokeWidth={2} maxBarSize={24} isAnimationActive={false} />
          <Bar dataKey="FairlyActiveMinutes" name="Fairly active" stackId="a" fill="var(--series-2)"
            stroke="var(--surface-1)" strokeWidth={2} maxBarSize={24} isAnimationActive={false} />
          <Bar dataKey="LightlyActiveMinutes" name="Lightly active" stackId="a" fill="var(--series-3)"
            stroke="var(--surface-1)" strokeWidth={2} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
        </BarChart>
      </ChartCard>

      <details className="card data-table wide">
        <summary>Daily data table</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Date</th><th>Steps</th><th>Calories</th><th>Asleep</th><th>Avg HR</th>
                <th>Very active</th><th>Fairly active</th><th>Lightly active</th><th>Sedentary</th>
              </tr>
            </thead>
            <tbody>
              {data.map(d => (
                <tr key={d.date}>
                  <td>{fmtDate(d.date)}</td>
                  <td>{fmtInt(d.TotalSteps)}</td>
                  <td>{fmtInt(d.Calories)}</td>
                  <td>{d.TotalMinutesAsleep != null ? fmtHours(d.TotalMinutesAsleep) : '—'}</td>
                  <td>{d.AverageHeartrate != null ? Math.round(d.AverageHeartrate) : '—'}</td>
                  <td>{d.VeryActiveMinutes}</td>
                  <td>{d.FairlyActiveMinutes}</td>
                  <td>{d.LightlyActiveMinutes}</td>
                  <td>{fmtInt(d.SedentaryMinutes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
