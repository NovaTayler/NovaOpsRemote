#!/usr/bin/env python3
"""
Quick start script for NovaOpsRemote in development mode
This script runs the application with minimal dependencies for testing
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, '.')

def check_basic_dependencies():
    """Check if basic dependencies are available"""
    required = ['fastapi', 'uvicorn', 'pydantic', 'structlog']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:", ', '.join(missing))
        print("Install with: pip install " + ' '.join(missing))
        return False
    
    return True

def main():
    print("🚀 NovaOpsRemote Quick Start")
    print("=" * 40)
    
    if not check_basic_dependencies():
        return False
    
    print("✅ Basic dependencies available")
    
    # Set development environment variables
    os.environ.setdefault("DB_URL", "postgresql://localhost:5432/novaops_dev")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    try:
        # Import and run the application
        print("📦 Loading application...")
        from main import app
        
        print("🌐 Starting server on http://localhost:8080")
        print("📚 API docs available at http://localhost:8080/docs")
        print("💡 Press Ctrl+C to stop")
        
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        return True
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("💡 Make sure PostgreSQL and Redis are running")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)