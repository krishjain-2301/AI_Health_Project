import axios from 'axios';

// Defaults to /api, which the Vite dev server proxies to the FastAPI server.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? '/api';

const http = axios.create({ baseURL: API_BASE });

export interface DayRow {
  date: string;
  TotalSteps: number;
  Calories: number;
  TotalDistance: number;
  TotalMinutesAsleep: number | null;
  TotalTimeInBed: number | null;
  SleepQualityScore: number | null;
  AverageHeartrate: number | null;
  MaxHeartrate: number | null;
  MinHeartrate: number | null;
  WeightKg: number | null;
  BMI: number | null;
  VeryActiveMinutes: number;
  FairlyActiveMinutes: number;
  LightlyActiveMinutes: number;
  SedentaryMinutes: number;
}

export interface DataQuality {
  total_days: number;
  days_worn: number;
  heart_rate_days: number;
  weight_days: number;
  sleep_days: number;
}

export interface HealthData {
  patient_id: number;
  time_series: DayRow[];
  metrics_change: Record<string, number>;
  data_quality: DataQuality;
}

export interface Driver {
  feature: string;
  label: string;
  unit: string;
  contribution_pct: number;
  patient_value: number;
  typical_value: number;
  placeholder: boolean;
}

export interface Anomaly {
  field: string;
  label: string;
  value: number;
  severity: 'low' | 'high';
  message: string;
}

export interface Forecast {
  predictions: { factor: string; risk: string; description: string }[];
  raw: {
    risk_probability_pct: number;
    predicted_state: string;
    risk_level: 'High' | 'Low';
    confidence_pct: number;
    anomalies: Anomaly[];
    explanation: Driver[];
    model_type: string;
    based_on: string[];
  };
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
  confusion_matrix: number[][];
  test_samples: number;
  size_kb?: number;
  latency_ms?: number;
  agreement_pct?: number;
}

export interface ModelInfo {
  label: { description: string; class_names: string[]; positive_class: string };
  features: string[];
  sequence_length: number;
  split: {
    method: string; folds: number;
    patients: number; windows: number;
    low_activity_share: number;
  };
  baselines: { majority: Metrics; persistence: Metrics };
  keras: Metrics;
  tflite: Metrics | Record<string, never>;
}

export interface ChatStatus { llm_enabled: boolean; provider: 'gemini' | 'claude' | null; model: string | null }
export interface ChatTurn { role: 'user' | 'assistant'; content: string }
export interface ChatReply { reply: string; source: 'gemini' | 'claude' | 'rules'; notice: string | null }

export const getPatients = () =>
  http.get<{ patients: number[] }>('/patients').then(r => r.data.patients);

export const getHealthData = (id: number) =>
  http.get<HealthData>(`/patients/${id}/health_data`).then(r => r.data);

export const getForecast = (id: number, useTinyML: boolean) =>
  http.post<Forecast>('/predict_consequences', { patient_id: id, use_tinyml: useTinyML }).then(r => r.data);

export const getModelInfo = () => http.get<ModelInfo>('/model_info').then(r => r.data);

export const getChatStatus = () => http.get<ChatStatus>('/chat/status').then(r => r.data);

export const sendChat = (message: string, role: string, patientId: number | null, history: ChatTurn[]) =>
  http.post<ChatReply>('/chat', {
    message, role, history, patient_context: { patient_id: patientId },
  }).then(r => r.data);

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    // The Vite proxy answers 500/502/504 itself when the API server is down
    if (!e.response || [500, 502, 504].includes(e.response.status) && !e.response.data?.detail)
      return "Can't reach the API server. Start it with `python main.py` in the server folder (port 8000).";
    return e.response.data?.detail ?? `The server returned an error (${e.response.status}).`;
  }
  return 'Something went wrong.';
}
