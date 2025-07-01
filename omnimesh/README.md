# OmniMesh Router Module

This module provides the enhanced API endpoints and WebSocket functionality for the OmniMesh distributed AI and e-commerce platform.

## Features

### Enhanced API Endpoints

- **`/api/status`** - System status with enhanced data including:
  - accounts: platform, email, status from `accounts` table
  - listings: sku, platform, status from `listings` table  
  - orders: order_id, platform, status from `orders` table
  - bots: bot_name, status, last_run from `bot_runs` table
  - Redis caching with 60-second TTL

- **`/api/health`** - Simple health check endpoint

### WebSocket Endpoints

- **`/ws/updates`** - Real-time system metrics streaming
- **`/api/ws/bots/{bot_name}`** - Bot log streaming via Redis pubsub

### Background Tasks

- **`ping_nodes()`** - Continuously pings nodes and updates Redis with system metrics
- **`update_metrics()`** - Updates global system metrics in Redis

## Installation

```bash
pip install fastapi uvicorn asyncpg redis psutil structlog
```

## Usage

### Basic Integration

```python
from fastapi import FastAPI
from omnimesh.router import router, start_background_tasks, stop_background_tasks

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    await start_background_tasks()

@app.on_event("shutdown") 
async def shutdown_event():
    await stop_background_tasks()
```

### Modern FastAPI Lifespan Integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from omnimesh.router import router, start_background_tasks, stop_background_tasks

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await start_background_tasks()
    yield
    # Shutdown
    await stop_background_tasks()

app = FastAPI(lifespan=lifespan)
app.include_router(router)
```

## Configuration

Set these environment variables:

```bash
DB_URL=postgresql://omnimesh:password@localhost:5432/omnimesh
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-jwt-secret
```

## Database Schema

The router expects these tables to exist:

```sql
-- Core OmniMesh tables
CREATE TABLE nodes (node_id VARCHAR(64), status VARCHAR(20), last_seen TIMESTAMP);
CREATE TABLE vaults (label VARCHAR(255), owner_node_id VARCHAR(64), created_at TIMESTAMP);
CREATE TABLE tasks (task_id VARCHAR(64), task_type VARCHAR(50), status VARCHAR(20));
CREATE TABLE blockchain_txns (tx_hash VARCHAR(66), from_address VARCHAR(42), status VARCHAR(20));
CREATE TABLE federated_rounds (round_id VARCHAR(64), model_id VARCHAR(64), status VARCHAR(20));
CREATE TABLE swarm_intelligence (swarm_id VARCHAR(64), status VARCHAR(20), created_at TIMESTAMP);

-- Enhanced e-commerce tables
CREATE TABLE accounts (platform TEXT, email TEXT, status TEXT);
CREATE TABLE listings (sku TEXT, platform TEXT, status TEXT);
CREATE TABLE orders (order_id TEXT, platform TEXT, status TEXT);
CREATE TABLE bot_runs (bot_name TEXT, status TEXT, last_run TIMESTAMP);
```

## API Examples

### Get System Status

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/status
```

Response includes:
```json
{
  "nodes": [...],
  "vaults": [...], 
  "tasks": [...],
  "accounts": [...],
  "listings": [...],
  "orders": [...],
  "bots": [...],
  "metrics": {
    "cpu_percent": 25.0,
    "memory_percent": 60.0,
    "disk_percent": 45.0
  }
}
```

### WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/updates');
ws.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  console.log('System metrics:', metrics);
};
```

### WebSocket Bot Logs

```javascript
const ws = new WebSocket('ws://localhost:8080/api/ws/bots/my-bot?token=<token>');
ws.onmessage = (event) => {
  console.log('Bot log:', event.data);
};
```

## Testing

Run the validation script:

```bash
python tests/validate_router.py
```

## Architecture

The router is designed to be:

- **Modular** - Can be integrated into existing FastAPI applications
- **Scalable** - Uses Redis caching and async operations
- **Real-time** - WebSocket support for live updates
- **Secure** - Token-based authentication (extensible with vault manager)
- **Observable** - Structured logging with contextual information

## Integration Notes

- The router uses dependency injection for authentication - replace `get_current_user` with your authentication system
- Background tasks are managed separately from the router to allow graceful shutdown
- Redis pubsub is used for bot log streaming - ensure your bot system publishes to `nova:logs_{bot_name}` channels
- Database connections are pooled for efficiency