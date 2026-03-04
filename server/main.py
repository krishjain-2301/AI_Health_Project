from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api import data_routes, ml_routes, chat_routes

app = FastAPI(title="AI Health Project API", description="Server for Health Analyzing Web Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5175", "http://127.0.0.1:5175", "http://localhost:5176", "http://127.0.0.1:5176"],
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
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
