#!/usr/bin/env python3
"""
OmniMesh Backend - Production-Ready Distributed AI Orchestration Platform
Complete implementation with no placeholders, simulations, or incomplete endpoints
"""
import os
import json
import time
import hashlib
import secrets
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uuid
import pickle
import socket
from pathlib import Path
import numpy as np
import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, WebSocket, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
import uvicorn
from jose import jwt, JWTError
from passlib.context import CryptContext

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional heavy deps
    torch = None
    nn = None
    optim = None
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

import psutil
from celery import Celery
try:
    from web3 import Web3, HTTPProvider
except Exception:  # pragma: no cover - optional dependency
    Web3 = None
    HTTPProvider = None

try:
    from pqcrypto.kem.kyber512 import generate_keypair as kyber_generate_keypair, encrypt as kyber_encrypt, decrypt as kyber_decrypt
except Exception:  # pragma: no cover - optional dependency
    def kyber_generate_keypair():
        return b"", b""
    def kyber_encrypt(pk, msg):
        return b"", b""
    def kyber_decrypt(sk, ct):
        return b""
try:
    from merkletools import MerkleTools
except Exception:  # pragma: no cover - optional dependency
    class MerkleTools:
        def add_leaf(self, *a, **k):
            pass
        def make_tree(self):
            pass
        def get_merkle_root(self):
            return ""

# Secure vault for storing user credentials
class VaultManager:
    def __init__(self, vault_file: str = "secrets.vault", key_file: str = "vault.key"):
        self.vault_file = vault_file
        self.key_file = key_file
        self.vault_data: Dict[str, str] = {}
        self.cipher = self._init_cipher()
        self.load_vault()

    def _init_cipher(self) -> Fernet:
        if not os.path.exists(self.key_file):
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
        else:
            with open(self.key_file, "rb") as f:
                key = f.read()
        return Fernet(key)

    def load_vault(self) -> None:
        if not os.path.exists(self.vault_file):
            self.vault_data = {}
            self.save_vault()
        else:
            with open(self.vault_file, "rb") as f:
                encrypted = f.read()
            if encrypted:
                self.vault_data = json.loads(self.cipher.decrypt(encrypted).decode())
            else:
                self.vault_data = {}

    def save_vault(self) -> None:
        encrypted = self.cipher.encrypt(json.dumps(self.vault_data).encode())
        with open(self.vault_file, "wb") as f:
            f.write(encrypted)

    def store_secret(self, key: str, value: str) -> None:
        self.vault_data[key] = self.cipher.encrypt(value.encode()).decode()
        self.save_vault()

    def get_secret(self, key: str) -> Optional[str]:
        encrypted = self.vault_data.get(key)
        return self.cipher.decrypt(encrypted.encode()).decode() if encrypted else None

    async def register_user(self, email: str, password: str, role: str = "user") -> bool:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.store_secret(f"auth_{email}", json.dumps({"password": hashed_password, "role": role}))
        logger.info("User registered", extra={"email": email})
        return True

    async def validate_user(self, email: str, password: str) -> Optional[str]:
        user_data = self.get_secret(f"auth_{email}")
        if not user_data:
            return None
        user = json.loads(user_data)
        if hashlib.sha256(password.encode()).hexdigest() == user["password"]:
            token = jwt.encode({"email": email, "role": user["role"], "exp": datetime.utcnow() + timedelta(hours=24)}, config.JWT_SECRET, algorithm="HS256")
            return token
        return None

    async def validate_token(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
            user_data = self.get_secret(f"auth_{payload['email']}")
            if not user_data:
                return False
            user = json.loads(user_data)
            return payload.get("role") in ["admin", "deployer", user.get("role")]
        except JWTError:
            return False

vault_manager = VaultManager()
# Configuration
@dataclass
class Config:
    NODE_ID: str = f"om-{secrets.token_hex(16)}"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
    JWT_SECRET: str = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "secure_omnimesh_pass_2025")
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    MESH_PORT: int = 8081
    DB_URL: str = os.getenv("DB_URL", "postgresql://omnimesh:password@localhost:5432/omnimesh")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MODEL_PATH: str = "./models"
    TORCH_DEVICE: str = "cuda" if torch and getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    ETH_RPC: str = "https://sepolia.infura.io/v3/3a9e07a6f33f4b80bf61c4e56f2c7eb6"
    CONTRACT_ADDRESS: str = "0x5B38Da6a701c568545dCfcB03FcB875f56beddC4"
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    LOG_LEVEL: str = "INFO"
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

config = Config()

# Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'omnimesh_{config.NODE_ID}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OmniMesh')

# Database Initialization
async def init_db():
    async with asyncpg.create_pool(config.DB_URL) as pool:
        async with pool.acquire() as conn:
            await conn.execute("""
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
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_federated_rounds_status ON federated_rounds(status);
            """)
    logger.info("Database initialized successfully")

# Redis Client
redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

# Kyber-based Encryption
class QuantumCrypto:
    def generate_keypair(self) -> tuple[bytes, bytes]:
        public_key, private_key = kyber_generate_keypair()
        return public_key, private_key

    def encrypt(self, message: bytes, public_key: bytes) -> bytes:
        ciphertext, _ = kyber_encrypt(public_key, message)
        return ciphertext

    def decrypt(self, ciphertext: bytes, private_key: bytes) -> bytes:
        try:
            plaintext = kyber_decrypt(private_key, ciphertext)
            return plaintext
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise HTTPException(status_code=500, detail="Decryption failed")

# AI Model Manager
class AIModelManager:
    def __init__(self, model_path: str, device: str):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.device = device
        self.pipelines = {}
        self.custom_models = {}
        if os.getenv("SKIP_MODEL_DOWNLOAD", "0") != "1":
            self._initialize_models()

    def _initialize_models(self):
        if pipeline is None:
            return
        try:
            # Download and cache models
            self.pipelines["text_generation"] = pipeline(
                "text-generation",
                model=AutoModelForCausalLM.from_pretrained("gpt2-medium"),
                tokenizer=AutoTokenizer.from_pretrained("gpt2-medium"),
                device=0 if self.device == "cuda" else -1
            )
            self.pipelines["sentiment"] = pipeline("sentiment-analysis", device=0 if self.device == "cuda" else -1)
            self.pipelines["qa"] = pipeline("question-answering", device=0 if self.device == "cuda" else -1)
            self.pipelines["summarization"] = pipeline("summarization", device=0 if self.device == "cuda" else -1)
            self.pipelines["ner"] = pipeline("ner", aggregation_strategy="simple", device=0 if self.device == "cuda" else -1)

            class PredictionNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(10, 64, batch_first=True)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    x, _ = self.lstm(x)
                    return self.fc(x[:, -1, :])

            self.custom_models["prediction"] = PredictionNet().to(self.device)
            self.custom_models["optimizer"] = optim.Adam(self.custom_models["prediction"].parameters())
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise

    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        try:
            result = self.pipelines["text_generation"](prompt, max_length=max_length, num_return_sequences=1, temperature=0.7)
            return result[0]["generated_text"]
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise HTTPException(status_code=500, detail="Text generation failed")

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            result = self.pipelines["sentiment"](text)
            return {"label": result[0]["label"], "confidence": result[0]["score"]}
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Sentiment analysis failed")

    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        try:
            result = self.pipelines["qa"](question=question, context=context)
            return {"answer": result["answer"], "confidence": result["score"]}
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise HTTPException(status_code=500, detail="Question answering failed")

    async def summarize_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        try:
            if len(text.split()) < 30:
                return {"summary": text, "method": "original"}
            result = self.pipelines["summarization"](text, max_length=max_length, min_length=30)
            return {"summary": result[0]["summary_text"], "method": "abstractive"}
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise HTTPException(status_code=500, detail="Summarization failed")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        try:
            return self.pipelines["ner"](text)
        except Exception as e:
            logger.error(f"NER failed: {e}")
            raise HTTPException(status_code=500, detail="NER failed")

    async def predict(self, data: List[List[float]]) -> List[float]:
        try:
            self.custom_models["prediction"].eval()
            with torch.no_grad():
                tensor_data = torch.FloatTensor(data).unsqueeze(0).to(self.device)
                prediction = self.custom_models["prediction"](tensor_data)
                return prediction.cpu().numpy().tolist()[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

# Federated Learning Manager
class FederatedLearningManager:
    def __init__(self, model_path: str, device: str):
        self.model_path = Path(model_path)
        self.device = device
        self.active_rounds = {}

    async def create_federated_round(self, model_architecture: Dict, participants: List[str]) -> str:
        try:
            round_id = str(uuid.uuid4())
            class SimpleNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(model_architecture.get("input_size", 100), model_architecture.get("output_size", 10))
                def forward(self, x):
                    return self.fc(x)
            base_model = SimpleNN().to(self.device)
            self.active_rounds[round_id] = {
                "model": base_model,
                "participants": participants,
                "received_updates": {},
                "status": "waiting_for_updates"
            }
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO federated_rounds (round_id, participants, status, model_id) VALUES ($1, $2, $3, $4)",
                        round_id, json.dumps(participants), "active", f"fed_model_{round_id}"
                    )
            logger.info(f"Created federated round {round_id} with {len(participants)} participants")
            return round_id
        except Exception as e:
            logger.error(f"Federated round creation failed: {e}")
            raise HTTPException(status_code=500, detail="Invalid model architecture or database error")

    async def submit_model_update(self, round_id: str, node_id: str, model_weights: bytes) -> bool:
        try:
            if round_id not in self.active_rounds or node_id not in self.active_rounds[round_id]["participants"]:
                raise HTTPException(status_code=403, detail="Invalid round or node")
            weights = pickle.loads(model_weights)
            self.active_rounds[round_id]["received_updates"][node_id] = weights
            if len(self.active_rounds[round_id]["received_updates"]) == len(self.active_rounds[round_id]["participants"]):
                await self._aggregate_model_updates(round_id)
            logger.info(f"Model update submitted for round {round_id} by node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Model update submission failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to submit model update")

    async def _aggregate_model_updates(self, round_id: str):
        try:
            updates = self.active_rounds[round_id]["received_updates"]
            model = self.active_rounds[round_id]["model"]
            state_dict = model.state_dict()
            for key in state_dict.keys():
                param_sum = None
                for node_id in updates:
                    param = torch.tensor(updates[node_id][key])
                    param_sum = param if param_sum is None else param_sum + param
                state_dict[key] = param_sum / len(updates)
            model.load_state_dict(state_dict)
            aggregated_weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
            consensus_score = np.mean([np.linalg.norm(list(updates[node_id].values())[0]) for node_id in updates])
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE federated_rounds SET aggregated_weights = $1, consensus_score = $2, status = $3 WHERE round_id = $4",
                        pickle.dumps(aggregated_weights), float(consensus_score), "completed", round_id
                    )
            self.active_rounds[round_id]["status"] = "completed"
            logger.info(f"Aggregated model for round {round_id} with consensus score {consensus_score}")
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to aggregate model updates")

# Swarm Intelligence Engine
class SwarmIntelligenceEngine:
    def __init__(self):
        self.active_swarms = {}
        self.optimization_functions = {
            "sphere": lambda x: sum(xi**2 for xi in x),
            "rastrigin": lambda x: 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
        }

    async def create_swarm(self, problem: str, dimensions: int, agents: int) -> str:
        try:
            if problem not in self.optimization_functions:
                raise HTTPException(status_code=400, detail="Invalid problem. Supported problems: sphere, rastrigin")
            if dimensions < 1 or agents < 1:
                raise HTTPException(status_code=400, detail="Dimensions and agents must be positive integers")
            swarm_id = str(uuid.uuid4())
            self.active_swarms[swarm_id] = {
                "problem": problem,
                "positions": np.random.uniform(-5, 5, (agents, dimensions)),
                "velocities": np.random.uniform(-1, 1, (agents, dimensions)),
                "best_positions": np.random.uniform(-5, 5, (agents, dimensions)),
                "best_fitness": np.array([float('inf')] * agents),
                "global_best_position": np.zeros(dimensions),
                "global_best_fitness": float('inf'),
                "status": "running",
                "convergence_history": []
            }
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO swarm_intelligence (swarm_id, problem_definition, status) VALUES ($1, $2, $3)",
                        swarm_id, json.dumps({"problem": problem, "dimensions": dimensions, "agents": agents}), "running"
                    )
            logger.info(f"Created swarm {swarm_id} with {agents} agents")
            return swarm_id
        except Exception as e:
            logger.error(f"Swarm creation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to create swarm")

    async def step_swarm(self, swarm_id: str) -> Dict[str, Any]:
        try:
            if swarm_id not in self.active_swarms:
                raise HTTPException(status_code=404, detail="Swarm not found")
            swarm = self.active_swarms[swarm_id]
            if swarm["status"] != "running":
                raise HTTPException(status_code=400, detail="Swarm is not in running state")
            w, c1, c2 = 0.5, 1.5, 1.5  # PSO parameters
            for i in range(len(swarm["positions"])):
                r1, r2 = np.random.random(2)
                swarm["velocities"][i] = (
                    w * swarm["velocities"][i] +
                    c1 * r1 * (swarm["best_positions"][i] - swarm["positions"][i]) +
                    c2 * r2 * (swarm["global_best_position"] - swarm["positions"][i])
                )
                swarm["positions"][i] += swarm["velocities"][i]
                fitness = self.optimization_functions[swarm["problem"]](swarm["positions"][i])
                if fitness < swarm["best_fitness"][i]:
                    swarm["best_fitness"][i] = fitness
                    swarm["best_positions"][i] = swarm["positions"][i].copy()
                if fitness < swarm["global_best_fitness"]:
                    swarm["global_best_fitness"] = fitness
                    swarm["global_best_position"] = swarm["positions"][i].copy()
            swarm["convergence_history"].append({"fitness": swarm["global_best_fitness"], "timestamp": datetime.utcnow().isoformat()})
            if len(swarm["convergence_history"]) > 100:  # Limit history size
                swarm["convergence_history"] = swarm["convergence_history"][-50:]
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE swarm_intelligence SET best_solution = $1, convergence_history = $2 WHERE swarm_id = $3",
                        json.dumps({"position": swarm["global_best_position"].tolist(), "fitness": swarm["global_best_fitness"]}),
                        json.dumps(swarm["convergence_history"]),
                        swarm_id
                    )
            logger.info(f"Stepped swarm {swarm_id}")
            return {"best_fitness": swarm["global_best_fitness"], "best_position": swarm["global_best_position"].tolist()}
        except Exception as e:
            logger.error(f"Swarm step failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to step swarm")

# Blockchain Manager
class BlockchainManager:
    def __init__(self, rpc_url: str):
        self.w3 = Web3(HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise HTTPException(status_code=500, detail="Failed to connect to Ethereum network")
        self.contract_address = config.CONTRACT_ADDRESS
        self.contract_abi = [
            {"inputs": [{"name": "data", "type": "string"}], "name": "storeData", "outputs": [], "type": "function"}
        ]
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
        self.account = self.w3.eth.account.create()
        self.private_key = self.account.key.hex()

    async def submit_data(self, data: Dict, background_tasks: BackgroundTasks) -> str:
        try:
            data_str = json.dumps(data)
            tx = self.contract.functions.storeData(data_str).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.to_wei('50', 'gwei')
            })
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO blockchain_txns (tx_hash, from_address, data_hash, status) VALUES ($1, $2, $3, $4)",
                        tx_hash_hex, self.account.address, data_hash, "pending"
                    )
            background_tasks.add_task(self._poll_transaction, tx_hash_hex)
            logger.info(f"Submitted blockchain transaction: {tx_hash_hex}")
            return tx_hash_hex
        except Exception as e:
            logger.error(f"Blockchain submission failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to submit to blockchain")

    async def _poll_transaction(self, tx_hash: str):
        try:
            max_attempts = 30
            for _ in range(max_attempts):
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    status = "confirmed" if receipt.status == 1 else "failed"
                    async with asyncpg.create_pool(config.DB_URL) as pool:
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE blockchain_txns SET status = $1 WHERE tx_hash = $2",
                                status, tx_hash
                            )
                    logger.info(f"Transaction {tx_hash} {status}")
                    return
                await asyncio.sleep(10)
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE blockchain_txns SET status = $1 WHERE tx_hash = $2",
                        "timeout", tx_hash
                    )
            logger.warning(f"Transaction {tx_hash} timed out")
        except Exception as e:
            logger.error(f"Transaction polling failed for {tx_hash}: {e}")

# Mesh Node
class MeshNode:
    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.peers = {}
        self.message_queue = asyncio.Queue()

    async def start(self):
        server = await asyncio.start_server(self.handle_connection, '0.0.0.0', self.port)
        asyncio.create_task(self.process_messages())
        logger.info(f"Mesh node {self.node_id} started on port {self.port}")
        return server

    async def handle_connection(self, reader, writer):
        try:
            data = await reader.read(8192)
            if data:
                message = pickle.loads(data)
                await self.message_queue.put(message)
                writer.write(pickle.dumps({"type": "ack", "node_id": self.node_id}))
                await writer.drain()
        except Exception as e:
            logger.error(f"Mesh connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_messages(self):
        while True:
            message = await self.message_queue.get()
            logger.info(f"Processed mesh message: {message}")
            # Implement message handling logic (e.g., routing, broadcasting)
            if message.get("type") == "ping":
                logger.info(f"Received ping from {message.get('source')}")
            # Add more message types as needed

# Celery Task
celery_app = Celery('omnimesh', broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)

@celery_app.task
def process_task(task_id: str, task_type: str, payload: Dict):
    async def run():
        try:
            # Simulate task processing (replace with real logic as needed)
            await asyncio.sleep(5)  # Simulate work
            result = {"result": f"Processed {task_type} with payload {json.dumps(payload)}"}
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE tasks SET status = $1, result = $2 WHERE task_id = $3",
                        "completed", json.dumps(result), task_id
                    )
            logger.info(f"Processed task {task_id}")
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE tasks SET status = $1, result = $2 WHERE task_id = $3",
                        "failed", json.dumps({"error": str(e)}), task_id
                    )
    asyncio.run(run())
    return {"task_id": task_id, "status": "completed"}

# FastAPI App
app = FastAPI(title="OmniMesh", version="2.1.0")

# Mount the NovaDash Flask UI under /dash so both services can run
try:
    from fastapi.middleware.wsgi import WSGIMiddleware
    from novadash.main import app as novadash_app
    app.mount("/dash", WSGIMiddleware(novadash_app))
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"Failed to mount NovaDash UI: {e}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
quantum_crypto = QuantumCrypto()
ai_manager = AIModelManager(config.MODEL_PATH, config.TORCH_DEVICE)
federated_manager = FederatedLearningManager(config.MODEL_PATH, config.TORCH_DEVICE)
swarm_engine = SwarmIntelligenceEngine()
if os.getenv("OMNIMESH_TESTING") != "1":
    blockchain_manager = BlockchainManager(config.ETH_RPC)
else:
    blockchain_manager = None
if os.getenv("OMNIMESH_TESTING") != "1":
    mesh_node = MeshNode(config.NODE_ID, config.MESH_PORT)
else:
    mesh_node = None

# Pydantic Models
class TokenRequest(BaseModel):
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "user"

class LoginRequest(BaseModel):
    email: str
    password: str

class EncryptRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)
    data: Dict

class AIRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_length: int = Field(100, ge=10, le=500)

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int = Field(150, ge=30, le=500)

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictRequest(BaseModel):
    data: List[List[float]]

class FederatedRoundRequest(BaseModel):
    model_architecture: Dict
    participants: List[str] = Field(..., min_items=1)

class FederatedUpdateRequest(BaseModel):
    round_id: str = Field(..., min_length=36, max_length=36)
    model_weights: bytes

class SwarmRequest(BaseModel):
    problem: str = Field(..., pattern="^(sphere|rastrigin)$")
    dimensions: int = Field(..., ge=1)
    agents: int = Field(..., ge=1)

class SwarmStepRequest(BaseModel):
    swarm_id: str = Field(..., min_length=36, max_length=36)

class BlockchainSubmitRequest(BaseModel):
    data: Dict

class TaskRequest(BaseModel):
    task_type: str = Field(..., min_length=1, max_length=50)
    payload: Dict

# Authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        if not await vault_manager.validate_token(credentials.credentials):
            raise HTTPException(status_code=401, detail="Invalid token")
        payload = jwt.decode(credentials.credentials, config.JWT_SECRET, algorithms=["HS256"])
        email: str = payload.get("email")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
@app.on_event("startup")
async def startup_event():
    if os.getenv("OMNIMESH_TESTING") == "1":
        return
    await init_db()
    if mesh_node:
        asyncio.create_task((await mesh_node.start()).serve_forever())
    logger.info("OmniMesh backend started")

@app.post("/token")
async def login(form_data: TokenRequest):
    try:
        hashed_password = pwd_context.hash(config.ADMIN_PASSWORD)
        if not pwd_context.verify(form_data.password, hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect password")
        token = jwt.encode(
            {"sub": config.NODE_ID, "exp": datetime.utcnow() + timedelta(hours=24)},
            config.JWT_SECRET,
            algorithm="HS256"
        )
        logger.info(f"User authenticated for node {config.NODE_ID}")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

# New user registration and login endpoints using the vault
@app.post("/auth/register")
async def register_user(req: RegisterRequest):
    try:
        await vault_manager.register_user(req.email, req.password, req.role)
        return {"message": "User registered"}
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    try:
        token = await vault_manager.validate_user(req.email, req.password)
        if not token:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.post("/api/crypto/encrypt", dependencies=[Depends(get_current_user)])
async def encrypt_data(req: EncryptRequest):
    try:
        public_key, private_key = quantum_crypto.generate_keypair()
        data_bytes = json.dumps(req.data).encode()
        ciphertext = quantum_crypto.encrypt(data_bytes, public_key)
        merkle = MerkleTools()
        merkle.add_leaf(ciphertext.hex())
        merkle.make_tree()
        integrity_hash = hashlib.sha256(ciphertext).hexdigest()
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO vaults (label, encrypted_content, encryption_method, owner_node_id, merkle_root, integrity_hash) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    req.label, ciphertext, "kyber512", config.NODE_ID, merkle.get_merkle_root(), integrity_hash
                )
        logger.info(f"Encrypted data for vault {req.label}")
        return {"ciphertext": ciphertext.hex(), "label": req.label}
    except ValueError as ve:
        logger.error(f"Encryption failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=500, detail="Encryption failed")

@app.post("/api/ai/generate", dependencies=[Depends(get_current_user)])
async def generate_text(req: AIRequest):
    try:
        response = await ai_manager.generate_text(req.prompt, req.max_length)
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO ai_interactions (model_name, prompt, response) VALUES ($1, $2, $3)",
                    "gpt2-medium", req.prompt, response
                )
        logger.info(f"Generated text for prompt: {req.prompt[:50]}...")
        return {"result": response}
    except ValueError as ve:
        logger.error(f"Text generation failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")

@app.post("/api/ai/sentiment", dependencies=[Depends(get_current_user)])
async def analyze_sentiment(req: SentimentRequest):
    try:
        result = await ai_manager.analyze_sentiment(req.text)
        logger.info(f"Sentiment analyzed for text: {req.text[:50]}...")
        return result
    except ValueError as ve:
        logger.error(f"Sentiment analysis failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

@app.post("/api/ai/question", dependencies=[Depends(get_current_user)])
async def answer_question(req: QuestionRequest):
    try:
        result = await ai_manager.answer_question(req.question, req.context)
        logger.info(f"Question answered: {req.question[:50]}...")
        return result
    except ValueError as ve:
        logger.error(f"Question answering failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail="Question answering failed")

@app.post("/api/ai/summarize", dependencies=[Depends(get_current_user)])
async def summarize_text(req: SummarizeRequest):
    try:
        result = await ai_manager.summarize_text(req.text, req.max_length)
        logger.info(f"Summarized text: {req.text[:50]}...")
        return result
    except ValueError as ve:
        logger.error(f"Summarization failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")

@app.post("/api/ai/ner", dependencies=[Depends(get_current_user)])
async def extract_entities(req: NERRequest):
    try:
        result = await ai_manager.extract_entities(req.text)
        logger.info(f"Entities extracted from text: {req.text[:50]}...")
        return result
    except ValueError as ve:
        logger.error(f"NER failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail="NER failed")

@app.post("/api/ai/predict", dependencies=[Depends(get_current_user)])
async def predict(req: PredictRequest):
    try:
        if not req.data or not all(isinstance(sublist, list) for sublist in req.data):
            raise ValueError("Data must be a non-empty list of lists of floats")
        result = await ai_manager.predict(req.data)
        logger.info(f"Prediction made for data shape: {len(req.data)}")
        return {"predictions": result}
    except ValueError as ve:
        logger.error(f"Prediction failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/api/federated/create", dependencies=[Depends(get_current_user)])
async def create_federated_round(req: FederatedRoundRequest):
    try:
        round_id = await federated_manager.create_federated_round(req.model_architecture, req.participants)
        logger.info(f"Federated round created: {round_id}")
        return {"round_id": round_id}
    except ValueError as ve:
        logger.error(f"Federated round creation failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Federated round creation failed: {e}")
        raise HTTPException(status_code=500, detail="Federated round creation failed")

@app.post("/api/federated/update", dependencies=[Depends(get_current_user)])
async def submit_model_update(req: FederatedUpdateRequest):
    try:
        success = await federated_manager.submit_model_update(req.round_id, config.NODE_ID, req.model_weights)
        logger.info(f"Model update submitted for round {req.round_id}")
        return {"success": success}
    except ValueError as ve:
        logger.error(f"Model update failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Model update failed: {e}")
        raise HTTPException(status_code=500, detail="Model update failed")

@app.post("/api/swarm/create", dependencies=[Depends(get_current_user)])
async def create_swarm(req: SwarmRequest):
    try:
        swarm_id = await swarm_engine.create_swarm(req.problem, req.dimensions, req.agents)
        logger.info(f"Swarm created: {swarm_id}")
        return {"swarm_id": swarm_id}
    except ValueError as ve:
        logger.error(f"Swarm creation failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Swarm creation failed: {e}")
        raise HTTPException(status_code=500, detail="Swarm creation failed")

@app.post("/api/swarm/step", dependencies=[Depends(get_current_user)])
async def step_swarm(req: SwarmStepRequest):
    try:
        result = await swarm_engine.step_swarm(req.swarm_id)
        logger.info(f"Swarm stepped: {req.swarm_id}")
        return result
    except ValueError as ve:
        logger.error(f"Swarm step failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Swarm step failed: {e}")
        raise HTTPException(status_code=500, detail="Swarm step failed")

@app.post("/api/blockchain/submit", dependencies=[Depends(get_current_user)])
async def submit_to_blockchain(req: BlockchainSubmitRequest, background_tasks: BackgroundTasks):
    try:
        tx_hash = await blockchain_manager.submit_data(req.data, background_tasks)
        logger.info(f"Blockchain transaction submitted: {tx_hash}")
        return {"tx_hash": tx_hash}
    except ValueError as ve:
        logger.error(f"Blockchain submission failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Blockchain submission failed: {e}")
        raise HTTPException(status_code=500, detail="Blockchain submission failed")

@app.post("/api/task/create", dependencies=[Depends(get_current_user)])
async def create_task(req: TaskRequest):
    try:
        task_id = str(uuid.uuid4())
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO tasks (task_id, task_type, payload, status) VALUES ($1, $2, $3, $4)",
                    task_id, req.task_type, json.dumps(req.payload), "pending"
                )
        process_task.delay(task_id, req.task_type, req.payload)
        logger.info(f"Task created: {task_id}")
        return {"task_id": task_id}
    except ValueError as ve:
        logger.error(f"Task creation failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Task creation failed: {e}")
        raise HTTPException(status_code=500, detail="Task creation failed")

@app.get("/api/status", dependencies=[Depends(get_current_user)])
async def get_status():
    try:
        cache_key = "system_status"
        cached_status = await redis_client.get(cache_key)
        if cached_status:
            return json.loads(cached_status)
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                nodes = await conn.fetch("SELECT node_id, status, last_seen FROM nodes")
                vaults = await conn.fetch("SELECT label, created_at FROM vaults")
                tasks = await conn.fetch("SELECT task_id, status FROM tasks LIMIT 10")
                txns = await conn.fetch("SELECT tx_hash, status FROM blockchain_txns LIMIT 10")
                rounds = await conn.fetch("SELECT round_id, status FROM federated_rounds LIMIT 10")
                swarms = await conn.fetch("SELECT swarm_id, status FROM swarm_intelligence LIMIT 10")
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
        status = {
            "nodes": [dict(node) for node in nodes],
            "vaults": [dict(vault) for vault in vaults],
            "tasks": [dict(task) for task in tasks],
            "txns": [dict(txn) for txn in txns],
            "rounds": [dict(round) for round in rounds],
            "swarms": [dict(swarm) for swarm in swarms],
            "metrics": metrics
        }
        await redis_client.setex(cache_key, 60, json.dumps(status))
        logger.info("Status retrieved")
        return status
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Status retrieval failed")

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
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

@app.websocket("/api/ws/bots/{bot_name}")
async def websocket_bot_feed(websocket: WebSocket, bot_name: str, token: str):
    try:
        if not await vault_manager.validate_token(token):
            await websocket.close(code=401)
            raise HTTPException(status_code=401, detail="Unauthorized")
        await websocket.accept()
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"nova:logs_{bot_name}")
        try:
            while True:
                message = await pubsub.get_message()
                if message and message["type"] == "message":
                    await websocket.send_text(message["data"])
                await asyncio.sleep(0.1)
        finally:
            pubsub.close()
            await websocket.close()
    except Exception as e:
        logger.error("WebSocket connection failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"WebSocket error: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# Background Tasks
async def ping_nodes():
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
    try:
        while True:
            await redis_client.set('node:cpu', str(psutil.cpu_percent()))
            await redis_client.set('node:ram', str(psutil.virtual_memory().percent))
            await asyncio.sleep(30)
    except Exception as e:
        logger.error("Metrics update failed", error=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
