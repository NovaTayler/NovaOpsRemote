"""
Pydantic models for dropshipping functionality
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class Product(BaseModel):
    title: str
    sku: str
    cost: float
    price: float
    url: str
    quantity: int
    supplier: str


class AccountInput(BaseModel):
    email: str
    password: str
    phone: str

    @validator('email')
    def email_valid(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v


class OrderFulfillRequest(BaseModel):
    order_id: str
    platform: str
    sku: str
    buyer_name: str
    buyer_address: str
    supplier: str


class BotDeployRequest(BaseModel):
    bot_name: str
    bot_path: str


class AccountCreationRequest(BaseModel):
    platform: str
    index: int


class ProductListingRequest(BaseModel):
    product: Product
    platform: str


class EmailAccount(BaseModel):
    email: str
    password: str
    status: str = "active"


class SupplierAccount(BaseModel):
    supplier: str
    email: str
    password: str
    api_key: Optional[str] = None
    net_terms: Optional[str] = None


class Listing(BaseModel):
    sku: str
    platform: str
    title: str
    price: float
    supplier: str
    status: str = "active"


class Order(BaseModel):
    order_id: str
    platform: str
    sku: str
    buyer_name: str
    buyer_address: str
    status: str = "pending"
    supplier: str
    fulfilled_at: Optional[str] = None


class PlatformAccount(BaseModel):
    platform: str
    email: str
    username: str
    password: str
    status: str = "active"
    token: Optional[str] = None