"""
Configuration management for NovaOpsRemote
Pydantic settings loading all environment variables
"""
import os
import secrets
from typing import List
from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Node identification
    NODE_ID: str = f"om-{secrets.token_hex(16)}"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
    JWT_SECRET: str = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "secure_omnimesh_pass_2025")
    
    # Network settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    MESH_PORT: int = 8081
    
    # Database settings
    DB_URL: str = os.getenv("DB_URL", "postgresql://omnimesh:password@localhost:5432/omnimesh")
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery settings
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    
    # AI/ML settings
    MODEL_PATH: str = "./models"
    TORCH_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Blockchain settings
    ETH_RPC: str = "https://sepolia.infura.io/v3/3a9e07a6f33f4b80bf61c4e56f2c7eb6"
    CONTRACT_ADDRESS: str = "0x5B38Da6a701c568545dCfcB03FcB875f56beddC4"
    
    # Application settings
    LOG_LEVEL: str = "INFO"
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # Dropshipping settings
    SUPPLIERS: List[str] = os.getenv("SUPPLIERS", "CJ Dropshipping,AliExpress").split(",")
    PLATFORMS: List[str] = ["eBay", "Amazon", "Walmart", "Etsy", "Shopify"]
    MAX_LISTINGS: int = int(os.getenv("MAX_LISTINGS", 500))
    
    # Email settings
    EMAIL_PROVIDER: str = os.getenv("EMAIL_PROVIDER", "imap.gmail.com")
    EMAIL_USER: str = os.getenv("EMAIL_USER", "")
    EMAIL_PASS: str = os.getenv("EMAIL_PASS", "")
    
    # Telegram settings
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


# Global configuration instance
config = Config()