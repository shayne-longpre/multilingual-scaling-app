#!/usr/bin/env python3
"""
Comprehensive test runner for scaling laws.
Run this script to validate all scaling law implementations.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_test_suite():
    """Run the complete test suite with reporting."""
    print("🧪 Running Scaling Laws Test Suite")
    print("=" * 50)
    
    test_files = [
        ("Property Tests", "tests/test_scaling_law_properties.py"),
        ("Integration Tests", "tests/test_app_integration.py"),
        ("Performance Tests", "tests/test_performance.py"),
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_file in test_files:
        print(f"\n📋 Running {test_name}...")
        start_time = time.time()
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--strict-markers"
            ], capture_output=True, text=True, timeout=300)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_name} PASSED ({duration:.1f}s)")
                results[test_name] = "PASSED"
            else:
                print(f"❌ {test_name} FAILED ({duration:.1f}s)")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results[test_name] = "FAILED"
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} TIMEOUT (>300s)")
            results[test_name] = "TIMEOUT"
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            results[test_name] = "ERROR"
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for test_name, status in results.items():
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "TIMEOUT": "⏰",
            "ERROR": "💥"
        }
        print(f"{status_emoji[status]} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    print(f"Duration: {total_duration:.1f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your scaling laws are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Check the output above.")
        return False


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("🚀 Running Quick Smoke Test...")
    
    try:
        from scaling_laws import ALL_SCALING_LAWS
        
        print(f"📚 Loaded {len(ALL_SCALING_LAWS)} scaling laws:")
        for name in ALL_SCALING_LAWS.keys():
            print(f"  - {name}")
        
        # Test basic functionality
        if "Chinchilla" in ALL_SCALING_LAWS:
            law = ALL_SCALING_LAWS["Chinchilla"].scaling_law
            
            # Test loss computation
            loss = law.loss(N=1e8, D=1e10)
            print(f"✅ Loss computation: {loss:.4f}")
            
            # Test optimization
            result = law.compute_optimal_allocation(C=1e20)
            print(f"✅ Optimization: N*={result['model']:.2e}, D*={result['data']:.2e}")
            
            print("🎯 Smoke test PASSED!")
            return True
        else:
            print("❌ Chinchilla law not found")
            return False
            
    except Exception as e:
        print(f"💥 Smoke test FAILED: {e}")
        return False


def main():
    """Main test runner."""
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        success = run_quick_smoke_test()
    else:
        success = run_test_suite()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()