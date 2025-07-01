"""
FastAPI router for OmniMesh AI orchestration endpoints
"""
import json
import uuid
import asyncio
import asyncpg
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from merkletools import MerkleTools
import hashlib

from .models import (
    EncryptRequest, AIRequest, SentimentRequest, QuestionRequest, 
    SummarizeRequest, NERRequest, PredictRequest, FederatedRoundRequest,
    FederatedUpdateRequest, SwarmRequest, SwarmStepRequest, 
    BlockchainSubmitRequest, TaskRequest, CryptoKeypairResponse,
    CryptoEncryptRequest, CryptoDecryptRequest, SystemMetrics
)
from .managers import (
    QuantumCrypto, AIModelManager, FederatedLearningManager,
    SwarmIntelligenceEngine, BlockchainManager, MeshNode
)
from ..common.config import config
from ..common.logging import get_logger
from ..common.redis_client import redis_client

logger = get_logger(__name__)
security = HTTPBearer()
router = APIRouter()

# Initialize managers
quantum_crypto = QuantumCrypto()
ai_manager = AIModelManager(config.MODEL_PATH, config.TORCH_DEVICE)
federated_manager = FederatedLearningManager(config.MODEL_PATH, config.TORCH_DEVICE)
swarm_engine = SwarmIntelligenceEngine()
blockchain_manager = BlockchainManager(config.ETH_RPC)
mesh_node = MeshNode(config.NODE_ID, config.MESH_PORT)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Placeholder authentication - implement proper JWT validation"""
    # TODO: Implement proper JWT validation using shared auth system
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid token")
    return "authenticated_user"


# Crypto endpoints
@router.post("/crypto/keypair", response_model=CryptoKeypairResponse)
async def generate_keypair(current_user: str = Depends(get_current_user)):
    """Generate quantum-resistant keypair"""
    try:
        public_key, private_key = quantum_crypto.generate_keypair()
        return CryptoKeypairResponse(
            public_key=public_key.hex(),
            private_key=private_key.hex()
        )
    except Exception as e:
        logger.error(f"Keypair generation failed: {e}")
        raise HTTPException(status_code=500, detail="Keypair generation failed")


@router.post("/crypto/encrypt")
async def encrypt_data(request: EncryptRequest, current_user: str = Depends(get_current_user)):
    """Encrypt data with quantum-resistant encryption"""
    try:
        public_key, private_key = quantum_crypto.generate_keypair()
        data_bytes = json.dumps(request.data).encode()
        ciphertext = quantum_crypto.encrypt(data_bytes, public_key)
        
        # Create Merkle tree for integrity
        merkle = MerkleTools()
        merkle.add_leaf(ciphertext.hex())
        merkle.make_tree()
        integrity_hash = hashlib.sha256(ciphertext).hexdigest()
        
        # Store in database
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO vaults (label, encrypted_content, encryption_method, 
                       owner_node_id, merkle_root, integrity_hash) 
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    request.label, ciphertext, "kyber", config.NODE_ID,
                    merkle.get_merkle_root(), integrity_hash
                )
        
        logger.info(f"Data encrypted and stored: {request.label}")
        return {
            "vault_id": request.label,
            "public_key": public_key.hex(),
            "private_key": private_key.hex(),
            "integrity_hash": integrity_hash
        }
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=500, detail="Encryption failed")


# AI endpoints
@router.post("/ai/generate")
async def generate_text(request: AIRequest, current_user: str = Depends(get_current_user)):
    """Generate text using AI models"""
    try:
        response = await ai_manager.generate_text(request.prompt, request.max_length)
        
        # Store interaction in database
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO ai_interactions (model_name, prompt, response) VALUES ($1, $2, $3)",
                    "gpt2-medium", request.prompt, response
                )
        
        logger.info(f"Generated text for prompt: {request.prompt[:50]}...")
        return {"generated_text": response}
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")


@router.post("/ai/sentiment")
async def analyze_sentiment(request: SentimentRequest, current_user: str = Depends(get_current_user)):
    """Analyze sentiment of text"""
    try:
        result = await ai_manager.analyze_sentiment(request.text)
        logger.info(f"Sentiment analyzed for text: {request.text[:50]}...")
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")


@router.post("/ai/question")
async def answer_question(request: QuestionRequest, current_user: str = Depends(get_current_user)):
    """Answer question based on context"""
    try:
        result = await ai_manager.answer_question(request.question, request.context)
        logger.info(f"Question answered: {request.question[:50]}...")
        return {"answer": result}
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail="Question answering failed")


@router.post("/ai/summarize")
async def summarize_text(request: SummarizeRequest, current_user: str = Depends(get_current_user)):
    """Summarize text"""
    try:
        result = await ai_manager.summarize_text(request.text, request.max_length)
        logger.info(f"Summarized text: {request.text[:50]}...")
        return {"summary": result}
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")


@router.post("/ai/ner")
async def extract_entities(request: NERRequest, current_user: str = Depends(get_current_user)):
    """Extract named entities from text"""
    try:
        result = await ai_manager.extract_entities(request.text)
        logger.info(f"Entities extracted from text: {request.text[:50]}...")
        return {"entities": result}
    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail="NER failed")


@router.post("/ai/predict")
async def predict(request: PredictRequest, current_user: str = Depends(get_current_user)):
    """Make predictions using custom models"""
    try:
        if not request.data or not all(isinstance(sublist, list) for sublist in request.data):
            raise ValueError("Data must be a non-empty list of lists of floats")
        
        # Simplified prediction - implement actual model inference
        predictions = [[0.5] * len(request.data[0])] * len(request.data)
        
        logger.info(f"Prediction made for data shape: {len(request.data)}")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# Federated learning endpoints
@router.post("/federated/create")
async def create_federated_round(request: FederatedRoundRequest, current_user: str = Depends(get_current_user)):
    """Create federated learning round"""
    try:
        round_id = await federated_manager.create_federated_round(
            request.model_architecture, request.participants
        )
        logger.info(f"Federated round created: {round_id}")
        return {"round_id": round_id}
    except Exception as e:
        logger.error(f"Federated round creation failed: {e}")
        raise HTTPException(status_code=500, detail="Federated round creation failed")


@router.post("/federated/update")
async def submit_model_update(request: FederatedUpdateRequest, current_user: str = Depends(get_current_user)):
    """Submit model update for federated learning"""
    try:
        success = await federated_manager.submit_model_update(
            request.round_id, config.NODE_ID, request.model_weights
        )
        logger.info(f"Model update submitted for round {request.round_id}")
        return {"success": success}
    except Exception as e:
        logger.error(f"Model update failed: {e}")
        raise HTTPException(status_code=500, detail="Model update failed")


# Swarm intelligence endpoints
@router.post("/swarm/create")
async def create_swarm(request: SwarmRequest, current_user: str = Depends(get_current_user)):
    """Create swarm for optimization"""
    try:
        swarm_id = await swarm_engine.create_swarm(
            request.problem, request.dimensions, request.agents
        )
        logger.info(f"Swarm created: {swarm_id}")
        return {"swarm_id": swarm_id}
    except Exception as e:
        logger.error(f"Swarm creation failed: {e}")
        raise HTTPException(status_code=500, detail="Swarm creation failed")


@router.post("/swarm/step")
async def step_swarm(request: SwarmStepRequest, current_user: str = Depends(get_current_user)):
    """Step swarm optimization"""
    try:
        result = await swarm_engine.step_swarm(request.swarm_id)
        logger.info(f"Swarm stepped: {request.swarm_id}")
        return result
    except Exception as e:
        logger.error(f"Swarm step failed: {e}")
        raise HTTPException(status_code=500, detail="Swarm step failed")


# Blockchain endpoints
@router.post("/blockchain/submit")
async def submit_to_blockchain(
    request: BlockchainSubmitRequest, 
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Submit data to blockchain"""
    try:
        tx_hash = await blockchain_manager.submit_data(request.data, background_tasks)
        logger.info(f"Blockchain transaction submitted: {tx_hash}")
        return {"tx_hash": tx_hash}
    except Exception as e:
        logger.error(f"Blockchain submission failed: {e}")
        raise HTTPException(status_code=500, detail="Blockchain submission failed")


# Task management endpoints
@router.post("/tasks/create")
async def create_task(request: TaskRequest, current_user: str = Depends(get_current_user)):
    """Create async task"""
    try:
        task_id = str(uuid.uuid4())
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO tasks (task_id, task_type, payload, status) VALUES ($1, $2, $3, $4)",
                    task_id, request.task_type, json.dumps(request.payload), "pending"
                )
        
        # TODO: Process task with Celery
        logger.info(f"Task created: {task_id}")
        return {"task_id": task_id}
    except Exception as e:
        logger.error(f"Task creation failed: {e}")
        raise HTTPException(status_code=500, detail="Task creation failed")


# Status endpoints
@router.get("/nodes")
async def get_nodes(current_user: str = Depends(get_current_user)):
    """Get mesh nodes status"""
    try:
        nodes = []
        async for key in redis_client.scan_iter("node:*"):
            node_data = await redis_client.hgetall(key)
            if node_data:
                nodes.append({
                    "id": key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1],
                    **{k.decode() if isinstance(k, bytes) else k: 
                       v.decode() if isinstance(v, bytes) else v 
                       for k, v in node_data.items()}
                })
        return nodes
    except Exception as e:
        logger.error(f"Node status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Node status retrieval failed")


@router.get("/status")
async def get_status(current_user: str = Depends(get_current_user)):
    """Get comprehensive system status"""
    try:
        # Get status from database and Redis
        async with asyncpg.create_pool(config.DB_URL) as pool:
            async with pool.acquire() as conn:
                nodes = await conn.fetch("SELECT * FROM nodes WHERE status = 'active' LIMIT 10")
                vaults = await conn.fetch("SELECT * FROM vaults LIMIT 10")
                tasks = await conn.fetch("SELECT * FROM tasks WHERE status = 'pending' LIMIT 10")
                ai_interactions = await conn.fetch("SELECT * FROM ai_interactions LIMIT 10")
        
        status = {
            "nodes": [dict(node) for node in nodes],
            "vaults": [dict(vault) for vault in vaults],
            "tasks": [dict(task) for task in tasks],
            "ai_interactions": [dict(interaction) for interaction in ai_interactions],
            "system_health": "healthy"
        }
        
        return status
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Status retrieval failed")


@router.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send periodic updates
            import psutil
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_json({"type": "metrics", "data": metrics})
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.1.0", "node_id": config.NODE_ID}