#!/usr/bin/env python3
"""
Example integration of OmniMesh router with FastAPI application
This shows how to wire the router with background tasks into the FastAPI lifecycle
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the OmniMesh router
from omnimesh.router import router, start_background_tasks, stop_background_tasks

# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle - startup and shutdown events
    """
    # Startup
    print("🚀 Starting OmniMesh application...")
    await start_background_tasks()
    print("✅ Background tasks started")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down OmniMesh application...")
    await stop_background_tasks()
    print("✅ Application shutdown complete")

# Create FastAPI application with lifespan management
app = FastAPI(
    title="OmniMesh - Distributed AI and E-Commerce Platform",
    description="Enhanced API with WebSocket support and background tasks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include the OmniMesh router
app.include_router(router)

# Optional: Add a root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OmniMesh API is running",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "health": "/api/health",
            "websocket_updates": "/ws/updates",
            "websocket_bot_logs": "/api/ws/bots/{bot_name}"
        }
    }

# Alternative approach for older FastAPI versions (without lifespan)
# @app.on_event("startup")
# async def startup_event():
#     """Start background tasks on application startup"""
#     await start_background_tasks()
# 
# @app.on_event("shutdown")
# async def shutdown_event():
#     """Stop background tasks on application shutdown"""
#     await stop_background_tasks()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "example_integration:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )