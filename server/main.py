import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
# Load server/.env (GEMINI_API_KEY etc.) before the routers read the environment.
# override=True: a key in this project's .env wins over one left in the system
# environment, which may be stale.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api import data_routes, ml_routes, chat_routes

_ready = threading.Event()

def _warm_up():
    """Load the dataset and both models once, in the background, so the first
    dashboard request does not pay for it. Requests that arrive early simply
    wait on the same locks."""
    try:
        from api import cnn_model, tinyml_model
        cnn_model.get_evaluation()
        tinyml_model.get_tflite_report()
    except Exception as e:
        print(f"[startup] Warm-up error: {e}")
    finally:
        _ready.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield

app = FastAPI(title="AI Health Project API", description="Server for Health Analyzing Web Application",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Any local dev port (Vite picks 5173, 5174, ... depending on what is free)
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_routes.router, prefix="/api")
app.include_router(ml_routes.router, prefix="/api")
app.include_router(chat_routes.router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Health Project Server Running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_ready": _ready.is_set()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
