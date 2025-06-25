#!/usr/bin/env python3
"""Run all test suites for the modular database system"""

import sys
import os
import subprocess
import time

def run_test_file(test_file):
    """Run a single test file and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING {test_file}")
    print('='*60)
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, test_file], 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, 
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if test passed
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            return True
        else:
            print(f"❌ {test_file} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {test_file} TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ {test_file} ERROR: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING COMPREHENSIVE TEST SUITE")
    print("Testing modular database system...")
    
    start_time = time.time()
    
    # List of test files in dependency order
    test_files = [
        "test_config.py",       # Configuration constants
        "test_models.py",       # Data models and exceptions
        "test_validators.py",   # Input validation
        "test_database_core.py", # Core database operations
        "test_embedding.py",    # AI embeddings and FAISS
        "test_search.py",       # Search functionality
        "test_filters.py",      # Filtering system
        "test_main.py",         # High-level interface integration
    ]
    
    # Run each test
    results = {}
    for test_file in test_files:
        results[test_file] = run_test_file(test_file)
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:<25} {status}")
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    print(f"⏱️  DURATION: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The modular system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)