# NovaOpsRemote

A unified platform for dropshipping automation and AI orchestration, combining the power of NovaShell's e-commerce automation with OmniMesh's distributed AI capabilities.

## 🏗️ Architecture

The application is structured as a modular FastAPI platform with three main packages:

### 📁 Directory Structure

```
NovaOpsRemote/
├── common/              # Shared utilities and services
│   ├── config.py        # Pydantic settings for all env vars
│   ├── db.py           # PostgreSQL connection pool
│   ├── redis_client.py # Redis client for caching/pub-sub
│   ├── celery_app.py   # Celery instance configuration
│   └── logging.py      # Structured logging setup
├── dropship/           # Dropshipping automation
│   ├── models.py       # Pydantic models for e-commerce
│   ├── router.py       # FastAPI endpoints (/dropship/*)
│   └── tasks.py        # Celery tasks for automation
├── omnimesh/           # AI orchestration platform
│   ├── models.py       # Pydantic models for AI operations
│   ├── router.py       # FastAPI endpoints (/omnimesh/api/*)
│   └── managers.py     # AI/Crypto/Blockchain managers
├── main.py             # Main FastAPI application
└── requirements.txt    # All dependencies
```

## 🚀 Features

### Dropshipping Automation (`/dropship`)
- **Account Creation**: Automated account creation on eBay, Amazon, Walmart, Etsy, Shopify
- **Product Sourcing**: Integration with CJ Dropshipping, AliExpress, and other suppliers
- **Listing Management**: Automated product listing with AI-generated descriptions
- **Order Fulfillment**: Automated order processing and supplier coordination
- **Payment Processing**: Secure payment handling with multiple providers

### AI Orchestration (`/omnimesh/api`)
- **Text Generation**: GPT-powered content creation
- **Sentiment Analysis**: Real-time text sentiment analysis
- **Question Answering**: Context-based AI responses
- **Named Entity Recognition**: Extract entities from text
- **Quantum Encryption**: Post-quantum cryptographic security
- **Federated Learning**: Distributed machine learning coordination
- **Swarm Intelligence**: Particle swarm optimization algorithms
- **Blockchain Integration**: Ethereum-based data integrity

### Shared Services (`/common`)
- **Configuration Management**: Environment-based settings
- **Database Pool**: AsyncPG PostgreSQL connection handling
- **Redis Cache**: Distributed caching and pub/sub messaging
- **Celery Tasks**: Distributed task queue processing
- **Structured Logging**: JSON-based logging with timestamps

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NovaTayler/NovaOpsRemote.git
   cd NovaOpsRemote
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the application**:
   ```bash
   python main.py
   ```

## 🌐 API Endpoints

### Root Endpoints
- `GET /` - Application info and service status
- `GET /health` - Health check with service status
- `GET /info` - Detailed application information
- `POST /auth/login` - User authentication
- `POST /auth/register` - User registration

### Dropshipping (`/dropship`)
- `POST /dropship/start_workflow` - Start automation workflow
- `POST /dropship/accounts/create` - Create platform accounts
- `GET /dropship/products` - Fetch products from suppliers
- `POST /dropship/products/list` - List products on platforms
- `POST /dropship/orders/fulfill` - Fulfill customer orders
- `POST /dropship/emails/create` - Create email accounts
- `POST /dropship/payments/process` - Process payments
- `GET /dropship/status` - Get automation status

### AI Orchestration (`/omnimesh/api`)
- `POST /omnimesh/api/crypto/keypair` - Generate quantum-safe keys
- `POST /omnimesh/api/crypto/encrypt` - Encrypt data
- `POST /omnimesh/api/ai/generate` - Generate text
- `POST /omnimesh/api/ai/sentiment` - Analyze sentiment
- `POST /omnimesh/api/ai/question` - Answer questions
- `POST /omnimesh/api/ai/summarize` - Summarize text
- `POST /omnimesh/api/ai/ner` - Extract named entities
- `POST /omnimesh/api/federated/create` - Create federated learning round
- `POST /omnimesh/api/swarm/create` - Create optimization swarm
- `POST /omnimesh/api/blockchain/submit` - Submit to blockchain
- `POST /omnimesh/api/tasks/create` - Create async tasks
- `GET /omnimesh/api/nodes` - Get mesh node status
- `GET /omnimesh/api/status` - Get system status
- `WS /omnimesh/api/ws/updates` - Real-time updates

## 🔐 Environment Variables

```bash
# Database
DB_URL=postgresql://user:pass@localhost:5432/novaops

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
ADMIN_PASSWORD=your-admin-password

# External APIs
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
CAPTCHA_API_KEY=your-2captcha-key
TWILIO_API_KEY=your-twilio-key

# Blockchain
ETH_RPC=https://sepolia.infura.io/v3/your-project-id

# Application
LOG_LEVEL=INFO
FRONTEND_URL=http://localhost:3000
```

## 🧪 Testing

Run the basic structure test:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from common.config import config
print(f'✅ NovaOpsRemote configured: {config.NODE_ID[:12]}...')
"
```

Or use the comprehensive test script:
```bash
# Create test script and run validation
python /tmp/simple_test.py
```

## 📊 Dependencies

### Core Framework
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server for production
- **Pydantic**: Data validation and settings

### Database & Cache
- **AsyncPG**: PostgreSQL async driver
- **Redis**: Caching and message broker
- **Celery**: Distributed task queue

### AI & ML
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face models
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning

### Blockchain & Crypto
- **Web3.py**: Ethereum interaction
- **PQCrypto**: Post-quantum cryptography
- **Cryptography**: Standard crypto operations

### Automation
- **Selenium**: Browser automation
- **Requests/aiohttp**: HTTP clients
- **BeautifulSoup**: HTML parsing

## 🔄 Deployment

### Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Set up services (PostgreSQL, Redis)
# Configure environment variables
# Run the application
uvicorn main:app --host 0.0.0.0 --port 8080
```

## 📈 Monitoring

The application provides comprehensive monitoring through:
- Health check endpoint (`/health`)
- Structured JSON logging
- Real-time WebSocket updates
- System metrics collection
- Database and Redis connection monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review the API documentation at `/docs` when running

---

**NovaOpsRemote v2.1.0** - Unified dropshipping automation and AI orchestration platform.