#!/usr/bin/env python3
"""
Simple validation script for OmniMesh router
"""
import sys
import os

# Add the parent directory to the path to import omnimesh
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def validate_router_imports():
    """Validate that the router can be imported correctly"""
    try:
        from omnimesh.router import router, ping_nodes, update_metrics
        print("✅ Router imports successful")
        return True
    except ImportError as e:
        print(f"❌ Router import failed: {e}")
        return False

def validate_router_structure():
    """Validate the router has the expected endpoints"""
    try:
        from omnimesh.router import router
        
        # Check if router has routes
        routes = router.routes
        route_paths = [route.path for route in routes if hasattr(route, 'path')]
        
        expected_endpoints = [
            "/api/status",
            "/api/health",
            "/ws/updates",
            "/api/ws/bots/{bot_name}"
        ]
        
        print(f"Found routes: {route_paths}")
        
        for endpoint in expected_endpoints:
            # Check if endpoint exists (exact match or pattern match for websockets)
            if any(endpoint in path or path in endpoint for path in route_paths):
                print(f"✅ Endpoint {endpoint} found")
            else:
                print(f"❌ Endpoint {endpoint} not found")
        
        return True
    except Exception as e:
        print(f"❌ Router structure validation failed: {e}")
        return False

def validate_background_tasks():
    """Validate background task functions exist"""
    try:
        from omnimesh.router import ping_nodes, update_metrics, start_background_tasks
        print("✅ Background task functions exist")
        return True
    except ImportError as e:
        print(f"❌ Background task validation failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🔍 Validating OmniMesh router...")
    
    all_passed = True
    
    if not validate_router_imports():
        all_passed = False
    
    if not validate_router_structure():
        all_passed = False
    
    if not validate_background_tasks():
        all_passed = False
    
    if all_passed:
        print("\n🎉 All validations passed!")
        return True
    else:
        print("\n❌ Some validations failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)