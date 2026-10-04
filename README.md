# HealthAI

A dashboard that turns Fitbit tracker data into a per-person health view: daily
trends, a next-day activity forecast with an explanation, a comparison of the
full model against its TinyML (edge) version, and a chat assistant for doctors
and patients.

It is a demo on a public research dataset, not a medical device.

## Showreel

[![HealthAI showreel: the dashboard, forecast, chat and model evaluation in 22 seconds](media/showreel.gif)](media/showreel.mp4)

A 22-second tour of the dashboard. Click it for the [full-quality video with sound](media/showreel.mp4).

## What is in here

| Folder | What it is |
|---|---|
| `Fitabase Data 4.12.16-5.12.16/` | The public Fitbit export: 33 people, 12 Apr – 12 May 2016 |
| `server/` | FastAPI backend: data merging, CNN, TFLite model, chat |
| `client/` | React + TypeScript + Vite dashboard |

## Running it

You need Python 3.12 and Node 20+.

**Server** (port 8000):

```bash
cd server
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The first start reads the CSVs (about 40 seconds) and trains the model. Both are
cached, so later starts take a few seconds.

**Client** (port 5173):

```bash
cd client
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend; set
`VITE_API_BASE` (see `client/.env.example`) to point somewhere else.

### Turning on AI answers for the chat

Put an API key in `server/.env` (see `server/.env.example`) and restart the
server: `GEMINI_API_KEY` for Google Gemini, or `ANTHROPIC_API_KEY` for Claude.
If both are set, Gemini is used. Without a key the chat still works, using
built-in keyword answers, and the sidebar says which mode it is in.

## How it works

1. **Data** (`server/api/dataset_merger.py`): joins daily activity with sleep,
   heart rate, weight and minute/hourly aggregates into one row per person per
   day. Only 14 of 33 people have heart-rate data and 8 have weight, so each row
   records whether those values were measured or filled with a placeholder. The
   dashboard and chat show "not recorded" rather than the placeholder.
2. **Forecast** (`server/api/cnn_model.py`): a small 1D CNN reads the last 3
   days the tracker was worn and gives the probability that the next day is a
   *low-activity day* (under 7,500 steps and under 30 very/fairly active
   minutes).
3. **Evaluation**: people are split into train / validation / test groups
   (21 / 5 / 7), so the reported scores come from people the model never saw.
   Scores are shown next to two simple baselines.
4. **Explanation**: for each forecast, every input is swapped for its typical
   value in turn; the change in probability is that input's contribution.
5. **TinyML** (`server/api/tinyml_model.py`): the CNN is converted to a
   quantized TFLite model, rebuilt whenever the CNN changes, and scored on the
   same test set for accuracy, size and speed.
6. **Chat** (`server/api/chat_routes.py`): Gemini or Claude answers from the
   selected person's data and forecast, with conversation history; falls back
   to rules when no key is set or the API call fails.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/patients` | List of person IDs |
| `GET /api/patients/{id}/health_data` | Daily series, change figures, data coverage |
| `POST /api/predict_consequences` | Forecast, explanation and out-of-range readings |
| `GET /api/model_info` | Label definition, split, held-out metrics, baselines, TFLite report |
| `GET /api/chat/status` | Which chat provider is configured |
| `POST /api/chat` | Chat reply (`source` is `gemini`, `claude` or `rules`) |

## Tests

```bash
cd server
python -m pytest tests
```

## Known limits

- The forecast is weak. On the held-out people it scores about the same as
  always guessing "Active"; three days of data from 21 people is not much to
  learn from. The dashboard shows this rather than hiding it.
- With 7 test people, the scores move noticeably with the random seed.
- The last day in the dataset is a partial day for most people, which is why the
  charts drop at the right edge.
