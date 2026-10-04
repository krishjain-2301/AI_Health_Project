import type { Metrics, ModelInfo } from '../api';

const pct = (v: number | undefined) => (v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);

export default function ModelPanel({ info, useTinyML }: { info: ModelInfo; useTinyML: boolean }) {
  const hasTflite = 'accuracy' in info.tflite;
  const tflite = hasTflite ? (info.tflite as Metrics) : null;
  const active = useTinyML && tflite ? tflite : info.keras;
  const activeName = useTinyML && tflite ? 'TinyML (TFLite)' : 'CNN (Keras)';
  const [[tn, fp], [fn, tp]] = active.confusion_matrix;
  const { split, baselines } = info;

  const rows: { name: string; m: Metrics; current?: boolean; muted?: boolean }[] = [
    { name: 'Baseline: always "Active"', m: baselines.majority, muted: true },
    { name: 'Baseline: same as today', m: baselines.persistence, muted: true },
    { name: 'CNN (Keras)', m: info.keras, current: !useTinyML },
    ...(tflite ? [{ name: 'TinyML (TFLite)', m: tflite, current: useTinyML }] : []),
  ];

  return (
    <section className="card model-panel">
      <div className="card-head">
        <div>
          <h2>How good is the forecast?</h2>
          <p className="sub">
            Scored on {split.test_windows} days from {split.test_patients} patients the model never saw in
            training (trained on {split.train_patients}, tuned on {split.val_patients}).
          </p>
        </div>
      </div>

      <div className="tiles four">
        {[
          { label: 'Accuracy', value: active.accuracy, hint: `always guessing "Active" scores ${pct(baselines.majority.accuracy)}` },
          { label: 'Precision', value: active.precision, hint: 'of days flagged low-activity, share that were' },
          { label: 'Recall', value: active.recall, hint: 'of real low-activity days, share that were caught' },
          { label: 'F1 score', value: active.f1, hint: 'balance of precision and recall' },
        ].map(t => (
          <div className="tile" key={t.label}>
            <div className="tile-label">{t.label}</div>
            <div className="tile-value">{pct(t.value)}</div>
            <div className="tile-hint">{t.hint}</div>
          </div>
        ))}
      </div>

      <div className="model-grid">
        <div>
          <h3>Confusion matrix — {activeName}</h3>
          <div className="cm">
            <div />
            <div className="cm-h">Predicted active</div>
            <div className="cm-h">Predicted low-activity</div>
            <div className="cm-h row">Was active</div>
            <div className="cm-cell hit"><b>{tn}</b>correct</div>
            <div className="cm-cell miss"><b>{fp}</b>false alarm</div>
            <div className="cm-h row">Was low-activity</div>
            <div className="cm-cell miss"><b>{fn}</b>missed</div>
            <div className="cm-cell hit"><b>{tp}</b>caught</div>
          </div>
        </div>

        <div>
          <h3>Against simple baselines, and on the edge</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Size</th><th>Speed</th></tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.name} className={`${r.current ? 'current' : ''} ${r.muted ? 'muted' : ''}`}>
                    <td>{r.name}</td>
                    <td>{pct(r.m.accuracy)}</td>
                    <td>{pct(r.m.precision)}</td>
                    <td>{pct(r.m.recall)}</td>
                    <td>{pct(r.m.f1)}</td>
                    <td>{r.m.size_kb !== undefined ? `${r.m.size_kb} KB` : '—'}</td>
                    <td>{r.m.latency_ms !== undefined ? `${r.m.latency_ms} ms` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {tflite && tflite.size_kb && info.keras.size_kb && (
            <p className="fine">
              The TinyML model is {(info.keras.size_kb / tflite.size_kb).toFixed(1)}× smaller than the CNN it
              was converted from and gives the same call on {tflite.agreement_pct}% of test days. Speed is
              the median time for one prediction on this machine.
            </p>
          )}
        </div>
      </div>

      <p className="fine definition">{info.label.description}</p>
    </section>
  );
}
