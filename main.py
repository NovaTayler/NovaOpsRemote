"""
NovaOpsRemote - Unified FastAPI application for dropshipping automation and OmniMesh AI orchestration
"""
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from passlib.context import CryptContext

# Import common modules
from common.config import config
from common.logging import setup_logging, get_logger

# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Import remaining common modules
from common.db import init_db, get_db_pool
from common.redis_client import redis_client
from common.celery_app import celery_app

# Import package routers
try:
    from dropship.router import router as dropship_router
    from omnimesh.router import router as omnimesh_router, mesh_node
    ROUTERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Router import failed: {e}")
    # Create mock routers for testing
    from fastapi import APIRouter
    dropship_router = APIRouter()
    omnimesh_router = APIRouter()
    mesh_node = None
    ROUTERS_AVAILABLE = False

# Global instances
db_pool = None
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events"""
    global db_pool
    
    # Startup
    logger.info("Starting NovaOpsRemote application")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Create database pool
        db_pool = await get_db_pool()
        logger.info("Database pool created")
        
        # Test Redis connection
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Start mesh node
        if ROUTERS_AVAILABLE and mesh_node:
            try:
                mesh_server = await mesh_node.start()
                if mesh_server:
                    logger.info(f"Mesh node started on port {config.MESH_PORT}")
            except Exception as e:
                logger.warning(f"Mesh node start failed: {e}")
        else:
            logger.info("Mesh node not available in testing mode")
        
        # TODO: Initialize Telegram bot if configured
        if config.TELEGRAM_BOT_TOKEN:
            logger.info("Telegram bot would be initialized here")
        
        logger.info(f"NovaOpsRemote started successfully on {config.HOST}:{config.PORT}")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down NovaOpsRemote application")
    
    try:
        if db_pool:
            await db_pool.close()
            logger.info("Database pool closed")
        
        await redis_client.close()
        logger.info("Redis connection closed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="NovaOpsRemote",
    description="Unified platform for dropshipping automation and AI orchestration",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Include routers with prefixes
app.include_router(dropship_router, prefix="/dropship", tags=["Dropshipping"])
app.include_router(omnimesh_router, prefix="/omnimesh/api", tags=["OmniMesh"])


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    uptime = time.time() - startup_time
    return {
        "message": "🚀 NovaOpsRemote is running!",
        "version": "2.1.0",
        "node_id": config.NODE_ID,
        "uptime": f"{uptime:.2f} seconds",
        "services": {
            "dropshipping": "/dropship",
            "omnimesh": "/omnimesh/api",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                db_healthy = True
        else:
            db_healthy = False
        
        # Test Redis connection
        try:
            await redis_client.ping()
            redis_healthy = True
        except:
            redis_healthy = False
        
        uptime = time.time() - startup_time
        
        health_status = {
            "status": "healthy" if db_healthy and redis_healthy else "degraded",
            "version": "2.1.0",
            "node_id": config.NODE_ID,
            "uptime": uptime,
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
                "celery": "unknown"  # TODO: Add Celery health check
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/info")
async def get_info():
    """Get application information"""
    return {
        "name": "NovaOpsRemote",
        "version": "2.1.0",
        "description": "Unified platform for dropshipping automation and AI orchestration",
        "node_id": config.NODE_ID,
        "features": [
            "Dropshipping automation",
            "AI model orchestration",
            "Quantum-resistant encryption",
            "Federated learning",
            "Swarm intelligence",
            "Blockchain integration",
            "Mesh networking"
        ],
        "endpoints": {
            "dropshipping": {
                "prefix": "/dropship",
                "main_endpoints": [
                    "/dropship/start_workflow",
                    "/dropship/accounts/create",
                    "/dropship/products",
                    "/dropship/products/list",
                    "/dropship/orders/fulfill"
                ]
            },
            "omnimesh": {
                "prefix": "/omnimesh/api",
                "main_endpoints": [
                    "/omnimesh/api/ai/generate",
                    "/omnimesh/api/crypto/encrypt",
                    "/omnimesh/api/federated/create",
                    "/omnimesh/api/swarm/create",
                    "/omnimesh/api/blockchain/submit"
                ]
            }
        }
    }


# Authentication endpoints (shared between services)
@app.post("/auth/login")
async def login(email: str, password: str):
    """Login endpoint - TODO: Implement proper authentication"""
    # This is a placeholder - implement proper JWT authentication
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    # TODO: Validate credentials against database
    # For now, return a mock token
    token = f"mock_token_for_{email}"
    
    logger.info(f"Login attempt for {email}")
    return {"token": token, "user": email}


@app.post("/auth/register")
async def register(email: str, password: str, role: str = "user"):
    """Register endpoint - TODO: Implement proper user registration"""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    # TODO: Store user in database with hashed password
    logger.info(f"Registration attempt for {email}")
    return {"message": "User registered successfully", "user": email}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )
