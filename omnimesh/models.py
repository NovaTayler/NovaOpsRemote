"""
Pydantic models for OmniMesh AI orchestration
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
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


class CryptoKeypairResponse(BaseModel):
    public_key: str
    private_key: str


class CryptoEncryptRequest(BaseModel):
    message: str
    public_key: str


class CryptoDecryptRequest(BaseModel):
    ciphertext: str
    private_key: str


class NodeInfo(BaseModel):
    node_id: str
    status: str = "active"
    last_seen: Optional[str] = None
    public_key: Optional[str] = None


class Vault(BaseModel):
    id: Optional[int] = None
    label: str
    encrypted_content: bytes
    encryption_method: str = "kyber"
    owner_node_id: str
    merkle_root: Optional[str] = None
    integrity_hash: Optional[str] = None


class AIInteraction(BaseModel):
    id: Optional[int] = None
    model_name: str
    prompt: str
    response: str


class BlockchainTransaction(BaseModel):
    id: Optional[int] = None
    tx_hash: str
    from_address: Optional[str] = None
    data_hash: str
    status: str = "pending"


class FederatedRound(BaseModel):
    id: Optional[int] = None
    round_id: str
    model_id: str
    participants: List[str]
    aggregated_weights: Optional[bytes] = None
    consensus_score: Optional[float] = None
    status: str = "active"


class SwarmIntelligence(BaseModel):
    id: Optional[int] = None
    swarm_id: str
    problem_definition: Dict
    best_solution: Optional[Dict] = None
    convergence_history: List[Dict] = []
    status: str = "running"


class Task(BaseModel):
    id: Optional[int] = None
    task_id: str
    task_type: str
    payload: Dict
    status: str = "pending"
    result: Optional[Dict] = None


class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    gpu_utilization: Optional[float] = None


class WebSocketMessage(BaseModel):
    type: str
    data: Dict
    timestamp: Optional[str] = None


class HealthStatus(BaseModel):
    status: str = "healthy"
    version: str = "2.1.0"
    node_id: str
    uptime: float