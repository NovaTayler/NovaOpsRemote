#!/usr/bin/env python3
"""
Complete integration example showing how to merge the OmniMesh router
with the existing integrated OmniMesh + NovaShell code
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer

# Add path for omnimesh module
sys.path.insert(0, os.path.dirname(__file__))

# Import the OmniMesh router
from omnimesh.router import router, start_background_tasks, stop_background_tasks

# Configuration class (simplified from the integrated code)
class Config:
    NODE_ID: str = "om-integrated-example"
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DB_URL: str = os.getenv("DB_URL", "postgresql://omnimesh:password@localhost:5432/omnimesh")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "fallback-secret")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

config = Config()

# Security
security = HTTPBearer()

# Simplified VaultManager placeholder (would be imported from integrated code)
class VaultManager:
    async def validate_token(self, token: str) -> bool:
        """Simplified token validation - replace with actual implementation"""
        return token and len(token) > 10

vault_manager = VaultManager()

# Enhanced authentication dependency for router
async def get_current_user_enhanced(credentials = Depends(security)):
    """Enhanced authentication that uses the vault manager"""
    if not credentials or not credentials.credentials:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Authorization required")
    
    if not await vault_manager.validate_token(credentials.credentials):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return credentials.credentials

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    print("🚀 Starting OmniMesh Integrated Application...")
    
    # Initialize database (would call init_db() from integrated code)
    print("📊 Initializing database...")
    
    # Start mesh node (would start mesh_node from integrated code)
    print("🌐 Starting mesh node...")
    
    # Start bot listener (would start bot_listener from integrated code)
    print("🤖 Starting bot listener...")
    
    # Start background tasks from the router
    await start_background_tasks()
    print("⚙️ Background tasks started")
    
    print("✅ Application startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application...")
    await stop_background_tasks()
    print("✅ Application shutdown complete")

# Create the main FastAPI application
app = FastAPI(
    title="OmniMesh + NovaShell Integrated Platform",
    description="Distributed AI and E-Commerce Platform with Enhanced Router",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Override the router's authentication with our enhanced version
# This is done by monkey patching the dependency in the router
import omnimesh.router
omnimesh.router.get_current_user = get_current_user_enhanced

# Include the enhanced OmniMesh router
app.include_router(router, tags=["OmniMesh Enhanced"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with integration information"""
    return {
        "message": "OmniMesh + NovaShell Integrated Platform",
        "version": "1.0.0",
        "node_id": config.NODE_ID,
        "features": [
            "Enhanced API endpoints with e-commerce data",
            "Real-time WebSocket metrics streaming", 
            "Bot log streaming via Redis pubsub",
            "Background node monitoring",
            "Integrated authentication"
        ],
        "endpoints": {
            "enhanced_status": "/api/status",
            "health": "/api/health", 
            "websocket_updates": "/ws/updates",
            "websocket_bot_logs": "/api/ws/bots/{bot_name}",
            "documentation": "/docs"
        }
    }

# Additional endpoints that would come from the integrated code
@app.post("/auth/login")
async def login():
    """Login endpoint (placeholder - would use integrated auth)"""
    return {"message": "Would integrate with vault manager authentication"}

@app.get("/api/nodes")
async def get_nodes():
    """Nodes endpoint (placeholder - would use integrated node management)"""
    return {"message": "Would integrate with mesh node management"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "integrated_example:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    )