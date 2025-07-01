"""
Celery tasks for dropshipping automation
"""
import os
import json
import random
import string
import asyncio
import aiohttp
import asyncpg
import base64
import imaplib
import email
import re
from typing import Dict, Optional, Tuple, List
from tenacity import retry, stop_after_attempt

# Conditional imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from fake_useragent import UserAgent
    FAKE_UA_AVAILABLE = True
except ImportError:
    FAKE_UA_AVAILABLE = False

from ..common.celery_app import celery_app
from ..common.config import config
from ..common.logging import get_logger
from ..common.redis_client import redis_client

logger = get_logger(__name__)

if FAKE_UA_AVAILABLE:
    ua = UserAgent()
else:
    ua = None


# Helper functions
async def get_random_user_agent() -> str:
    if FAKE_UA_AVAILABLE and ua:
        return ua.random
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


async def generate_email() -> str:
    domain = os.getenv("DOMAIN", "gmail.com")
    user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"{user}@{domain}"


async def solve_captcha(site_key: str, url: str) -> Optional[str]:
    async with aiohttp.ClientSession() as session:
        captcha_url = "http://2captcha.com/in.php"
        params = {
            "key": os.getenv("CAPTCHA_API_KEY"), 
            "method": "userrecaptcha", 
            "googlekey": site_key, 
            "pageurl": url
        }
        async with session.post(captcha_url, data=params) as resp:
            text = await resp.text()
            if "OK" not in text:
                return None
            captcha_id = text.split("|")[1]
            for _ in range(10):
                await asyncio.sleep(5)
                async with session.get(
                    f"http://2captcha.com/res.php?key={os.getenv('CAPTCHA_API_KEY')}&action=get&id={captcha_id}"
                ) as resp:
                    text = await resp.text()
                    if "OK" in text:
                        return text.split("|")[1]
            return None


async def get_virtual_phone() -> str:
    twilio_key = os.getenv("TWILIO_API_KEY")
    if not twilio_key:
        return f"+1555{random.randint(1000000, 9999999)}"
    
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Basic {base64.b64encode(twilio_key.encode()).decode()}"}
    ) as session:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_key.split(':')[0]}/IncomingPhoneNumbers.json"
        async with session.post(url, data={"AreaCode": "555"}) as resp:
            if resp.status == 201:
                return (await resp.json())["phone_number"]
            return f"+1555{random.randint(1000000, 9999999)}"


async def human_like_typing(element, text: str):
    """Simulate human typing"""
    if not SELENIUM_AVAILABLE:
        # Mock implementation
        return
    
    for char in text:
        element.send_keys(char)
        await asyncio.sleep(random.uniform(0.05, 0.3))


# Core dropshipping tasks
@celery_app.task(bind=True)
@retry(stop=stop_after_attempt(5))
def create_platform_account_task(self, platform: str, index: int) -> Dict:
    """Create account on e-commerce platform"""
    async def run():
        try:
            email = await generate_email()
            username = f"{platform.lower()}user{index}{random.randint(100, 999)}"
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            phone = await get_virtual_phone()
            
            signup_urls = {
                "eBay": "https://signup.ebay.com/pa/register",
                "Amazon": "https://sellercentral.amazon.com/register",
                "Walmart": "https://marketplace.walmart.com/us/seller-signup",
                "Etsy": "https://www.etsy.com/sell",
                "Shopify": "https://www.shopify.com/signup"
            }
            
            token = None
            
            # Simplified account creation logic
            # In production, this would use platform-specific APIs or automation
            if platform in signup_urls:
                # Store account info in database
                async with asyncpg.create_pool(config.DB_URL) as pool:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """INSERT INTO accounts (platform, email, username, password, status, token) 
                               VALUES ($1, $2, $3, $4, $5, $6)""",
                            platform, email, username, password, "created", token
                        )
                        
                logger.info(f"Created account on {platform}", email=email, username=username)
                return {"username": username, "email": email, "token": token}
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"Platform account creation failed: {e}")
            raise self.retry(exc=e)
    
    return asyncio.run(run())


@celery_app.task
@retry(stop=stop_after_attempt(3))
def fetch_products_task() -> List[Dict]:
    """Fetch products from suppliers"""
    async def run():
        suppliers = config.SUPPLIERS
        all_products = []
        
        for supplier in suppliers:
            cached = await redis_client.get(f"products:{supplier}")
            if cached:
                all_products.extend(json.loads(cached))
                continue
                
            # Simplified product fetching - implement real API calls
            products = []
            
            # Store in cache
            await redis_client.setex(f"products:{supplier}", 3600, json.dumps(products))
            all_products.extend(products)
            
        return all_products
    
    return asyncio.run(run())


@celery_app.task
@retry(stop=stop_after_attempt(3))
def list_product_task(product: Dict, platform: str) -> bool:
    """List product on e-commerce platform"""
    async def run():
        try:
            # This would use platform APIs to list products
            # Simplified implementation
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO listings (sku, platform, title, price, supplier, status) 
                           VALUES ($1, $2, $3, $4, $5, $6)""",
                        product["sku"], platform, product["title"], 
                        product["price"], product["supplier"], "active"
                    )
                    
            logger.info(f"Listed product on {platform}", sku=product["sku"])
            return True
            
        except Exception as e:
            logger.error(f"Product listing failed: {e}")
            raise
    
    return asyncio.run(run())


@celery_app.task
@retry(stop=stop_after_attempt(3))
def fulfill_order_task(order_id: str, platform: str, sku: str, buyer_name: str, 
                      buyer_address: str, supplier: str) -> bool:
    """Fulfill order with supplier"""
    async def run():
        try:
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    # Check if listing exists
                    listing = await conn.fetchrow(
                        "SELECT * FROM listings WHERE sku = $1 AND platform = $2", 
                        sku, platform
                    )
                    if not listing:
                        raise Exception("Listing not found")
                    
                    # Update order status
                    await conn.execute(
                        """INSERT INTO orders (order_id, platform, sku, buyer_name, buyer_address, status, supplier) 
                           VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                        order_id, platform, sku, buyer_name, buyer_address, "fulfilled", supplier
                    )
                    
            logger.info(f"Fulfilled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order fulfillment failed: {e}")
            raise
    
    return asyncio.run(run())


@celery_app.task
def create_email_account_task(provider: str, count: int = 1) -> List[Dict]:
    """Create email accounts for automation"""
    async def run():
        accounts = []
        for i in range(count):
            email = await generate_email()
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            
            # Store account
            async with asyncpg.create_pool(config.DB_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO accounts (platform, email, username, password, status) 
                           VALUES ($1, $2, $3, $4, $5)""",
                        "email", email, email.split("@")[0], password, "active"
                    )
                    
            accounts.append({"email": email, "password": password})
            
        return accounts
    
    return asyncio.run(run())


@celery_app.task 
def process_payment_task(order_id: str, amount: float, payment_method: str) -> Dict:
    """Process payment for order"""
    async def run():
        try:
            # Simplified payment processing
            # In production, integrate with actual payment processors
            
            result = {
                "order_id": order_id,
                "amount": amount,
                "payment_method": payment_method,
                "status": "completed",
                "transaction_id": f"txn_{random.randint(100000, 999999)}"
            }
            
            logger.info(f"Processed payment for order {order_id}", amount=amount)
            return result
            
        except Exception as e:
            logger.error(f"Payment processing failed: {e}")
            raise
    
    return asyncio.run(run())