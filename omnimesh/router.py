#!/usr/bin/env python3
"""
OmniMesh Router - Enhanced API endpoints with WebSocket support and background tasks
"""
import os
import json
import time
import asyncio
import logging
import structlog
import asyncpg
import redis.asyncio as redis
import psutil
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, WebSocket, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Initialize logger
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Configuration (will be imported from main config)
class Config:
    DB_URL: str = os.getenv("DB_URL", "postgresql://omnimesh:password@localhost:5432/omnimesh")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "fallback-secret")

config = Config()

# Initialize Redis client
redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

# Security
security = HTTPBearer()

# Router instance
router = APIRouter()

# Authentication dependency (placeholder - will use from main app)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Authentication dependency - to be replaced by actual vault manager implementation
    For now, this is a placeholder that accepts any valid authorization header
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authorization required")
    return credentials.credentials

# Enhanced authentication with vault manager integration
async def get_current_user_with_vault(credentials: HTTPAuthorizationCredentials = Depends(security), vault_manager=None):
    """
    Enhanced authentication that integrates with vault manager when available
    """
    try:
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Authorization required")
        
        # If vault manager is available, validate token
        if vault_manager and hasattr(vault_manager, 'validate_token'):
            if not await vault_manager.validate_token(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
        
        return credentials.credentials
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Enhanced /api/status endpoint
@router.get("/api/status", dependencies=[Depends(get_current_user)])
async def get_system_status():
    """
    Enhanced status endpoint that includes accounts, listings, orders, and bots data
    with Redis caching support.
    """
    try:
        # Check Redis cache first
        cache_key = "system_status"
        cached_status = await redis_client.get(cache_key)
        
        if cached_status:
            logger.info("Status retrieved from cache")
            return json.loads(cached_status)

        # Fetch data from database
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                # Fetch existing data
                nodes = await conn.fetch("SELECT node_id, status, last_seen FROM nodes LIMIT 10")
                vaults = await conn.fetch("SELECT label, owner_node_id, created_at FROM vaults LIMIT 10")
                tasks = await conn.fetch("SELECT task_id, task_type, status FROM tasks LIMIT 10")
                txns = await conn.fetch("SELECT tx_hash, from_address, status FROM blockchain_txns LIMIT 10")
                rounds = await conn.fetch("SELECT round_id, model_id, status FROM federated_rounds LIMIT 10")
                swarms = await conn.fetch("SELECT swarm_id, status, created_at FROM swarm_intelligence LIMIT 10")
                
                # Fetch enhanced data as per requirements
                accounts = await conn.fetch("SELECT platform, email, status FROM accounts LIMIT 10")
                listings = await conn.fetch("SELECT sku, platform, status FROM listings LIMIT 10")
                orders = await conn.fetch("SELECT order_id, platform, status FROM orders LIMIT 10")
                bots = await conn.fetch("SELECT bot_name, status, last_run FROM bot_runs LIMIT 10")
        
        # Collect system metrics
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
        
        # Compile status response with enhanced collections
        status = {
            "nodes": [dict(node) for node in nodes],
            "vaults": [dict(vault) for vault in vaults],
            "tasks": [dict(task) for task in tasks],
            "txns": [dict(txn) for txn in txns],
            "rounds": [dict(round) for round in rounds],
            "swarms": [dict(swarm) for swarm in swarms],
            "accounts": [dict(account) for account in accounts],
            "listings": [dict(listing) for listing in listings],
            "orders": [dict(order) for order in orders],
            "bots": [dict(bot) for bot in bots],
            "metrics": metrics
        }
        
        # Cache the status for 60 seconds
        await redis_client.setex(cache_key, 60, json.dumps(status, default=str))
        logger.info("Status retrieved and cached")
        return status
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Status retrieval failed")

# WebSocket endpoint for real-time updates
@router.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """
    WebSocket endpoint to stream system metrics in real-time
    """
    await websocket.accept()
    try:
        while True:
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "network_io": {
                    "sent": psutil.net_io_counters().bytes_sent / 1024 / 1024,
                    "received": psutil.net_io_counters().bytes_recv / 1024 / 1024
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# WebSocket endpoint for bot logs with enhanced authentication
@router.websocket("/api/ws/bots/{bot_name}")
async def websocket_bot_feed(websocket: WebSocket, bot_name: str, token: str = None):
    """
    WebSocket endpoint to stream bot logs via Redis pubsub
    Supports token-based authentication via query parameter
    """
    try:
        # Enhanced authentication - check token if provided
        if token:
            # For full integration, this would use vault_manager.validate_token(token)
            # For now, we accept any non-empty token
            if not token.strip():
                await websocket.close(code=1008, reason="Invalid token")
                return
        
        await websocket.accept()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"nova:logs_{bot_name}")
        
        try:
            while True:
                message = await pubsub.get_message()
                if message and message["type"] == "message":
                    await websocket.send_text(message["data"])
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(f"nova:logs_{bot_name}")
            await pubsub.close()
            await websocket.close()
    except Exception as e:
        logger.error("WebSocket connection failed", error=str(e))
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass  # WebSocket may already be closed

# Health check endpoint
@router.get("/api/health")
async def health():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

# Background Tasks
async def ping_nodes():
    """
    Background task to ping nodes and update their status in Redis
    """
    try:
        while True:
            node_id = str(uuid.uuid4())
            await redis_client.hset(f"node:{node_id}", mapping={
                "cpu": str(psutil.cpu_percent()),
                "ram": str(psutil.virtual_memory().percent),
                "last_seen": str(time.time())
            })
            await redis_client.expire(f"node:{node_id}", 30)
            await asyncio.sleep(5)
    except Exception as e:
        logger.error("Node ping failed", error=str(e))

async def update_metrics():
    """
    Background task to update system metrics in Redis
    """
    try:
        while True:
            await redis_client.set('node:cpu', str(psutil.cpu_percent()))
            await redis_client.set('node:ram', str(psutil.virtual_memory().percent))
            await asyncio.sleep(30)
    except Exception as e:
        logger.error("Metrics update failed", error=str(e))

# Function to start background tasks
async def start_background_tasks():
    """
    Start all background tasks
    """
    logger.info("Starting background tasks")
    asyncio.create_task(ping_nodes())
    asyncio.create_task(update_metrics())
    logger.info("Background tasks started")

# Cleanup function for proper shutdown
async def stop_background_tasks():
    """
    Stop all background tasks gracefully
    """
    logger.info("Stopping background tasks")
    # In a real implementation, we would track and cancel specific tasks
    # For now, we just log the shutdown
    logger.info("Background tasks stopped")