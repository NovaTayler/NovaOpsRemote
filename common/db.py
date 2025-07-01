"""
Database connection and initialization for NovaOpsRemote
One asyncpg pool initializer for shared Postgres database
"""
import asyncpg
from typing import Optional
from .config import config
from .logging import get_logger

logger = get_logger(__name__)


async def get_db_pool():
    """Create and return a database connection pool"""
    return await asyncpg.create_pool(config.DB_URL)


async def init_db():
    """Initialize database tables for both OmniMesh and dropshipping"""
    async with asyncpg.create_pool(config.DB_URL) as pool:
        async with pool.acquire() as conn:
            await conn.execute("""
                -- OmniMesh tables
                CREATE TABLE IF NOT EXISTS nodes (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(64) UNIQUE NOT NULL,
                    public_key TEXT,
                    status VARCHAR(20) DEFAULT 'active',
                    last_seen TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS vaults (
                    id SERIAL PRIMARY KEY,
                    label VARCHAR(255) NOT NULL,
                    encrypted_content BYTEA NOT NULL,
                    encryption_method VARCHAR(50),
                    owner_node_id VARCHAR(64),
                    merkle_root VARCHAR(64),
                    integrity_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS ai_interactions (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255),
                    prompt TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS blockchain_txns (
                    id SERIAL PRIMARY KEY,
                    tx_hash VARCHAR(66) UNIQUE NOT NULL,
                    from_address VARCHAR(42),
                    data_hash VARCHAR(64),
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS federated_rounds (
                    id SERIAL PRIMARY KEY,
                    round_id VARCHAR(64) UNIQUE NOT NULL,
                    model_id VARCHAR(64),
                    participants JSONB,
                    aggregated_weights BYTEA,
                    consensus_score FLOAT,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS swarm_intelligence (
                    id SERIAL PRIMARY KEY,
                    swarm_id VARCHAR(64) UNIQUE NOT NULL,
                    problem_definition JSONB,
                    best_solution JSONB,
                    convergence_history JSONB DEFAULT '[]',
                    status VARCHAR(20) DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(64) UNIQUE NOT NULL,
                    task_type VARCHAR(50),
                    payload JSONB,
                    status VARCHAR(20) DEFAULT 'pending',
                    result JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                -- Dropshipping tables
                CREATE TABLE IF NOT EXISTS accounts (
                    platform TEXT,
                    email TEXT PRIMARY KEY,
                    username TEXT,
                    password TEXT,
                    status TEXT,
                    token TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS supplier_accounts (
                    supplier TEXT,
                    email TEXT,
                    password TEXT,
                    api_key TEXT,
                    net_terms TEXT,
                    PRIMARY KEY (supplier, email)
                );
                CREATE TABLE IF NOT EXISTS listings (
                    sku TEXT PRIMARY KEY,
                    platform TEXT,
                    title TEXT,
                    price REAL,
                    supplier TEXT,
                    status TEXT
                );
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    platform TEXT,
                    sku TEXT,
                    buyer_name TEXT,
                    buyer_address TEXT,
                    status TEXT,
                    supplier TEXT,
                    fulfilled_at TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    password TEXT,
                    role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS bot_runs (
                    run_id TEXT PRIMARY KEY,
                    bot_name TEXT,
                    execution_time REAL,
                    errors INTEGER,
                    last_run TIMESTAMP,
                    status TEXT
                );
                
                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_federated_rounds_status ON federated_rounds(status);
            """)
    logger.info("Database initialized successfully")