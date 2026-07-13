"""
api/main.py

FastAPI Application initialization and entrypoint.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.config import APIConfig
from api.dependencies import init_engines
from api.routes import router

# Setup API logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("insider_threat.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager to initialize ML models and registries on startup."""
    logger.info("Starting up Insider Threat REST API...")
    config = APIConfig()
    try:
        init_engines(config)
        logger.info("REST API initialization complete. All ML components loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize REST API ML components: {e}")
        # We don't raise here to allow API to boot into an unhealthy/informative state rather than crash
        
    yield
    
    logger.info("Shutting down REST API...")

app = FastAPI(
    title="Insider Threat Detection System REST API",
    description="Production-grade real-time inference, risk assessment, and explainability REST API.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for SOC dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers
app.include_router(router)

# Custom global exception handler for standard runtime issues
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception in API endpoint: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error_type": type(exc).__name__,
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    config = APIConfig()
    logger.info(f"Launching Uvicorn server on http://{config.host}:{config.port} ...")
    uvicorn.run("api.main:app", host=config.host, port=config.port, reload=False)
