#!/usr/bin/env python3
import os
import platform
import subprocess
import sys

# ----------------------------
# 1. Create virtual environment
# ----------------------------
venv_dir = "venv"

if not os.path.exists(venv_dir):
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
else:
    print("Virtual environment already exists.")

# Determine activate script
if platform.system() == "Windows":
    activate = os.path.join(venv_dir, "Scripts", "activate.bat")
    python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
    mpiexec_bin = "mpiexec"
else:
    activate = os.path.join(venv_dir, "bin", "activate")
    python_bin = os.path.join(venv_dir, "bin", "python")
    mpiexec_bin = "mpiexec"

print(f"Using Python from virtualenv: {python_bin}")

# ----------------------------
# 2. Upgrade pip
# ----------------------------
subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

# ----------------------------
# 3. Install package with extras
# ----------------------------
subprocess.check_call([python_bin, "-m", "pip", "install", "-e", ".[test,optional]"])

# ----------------------------
# 4. Run all stats tests
# ----------------------------
print("\nRunning stats tests...")
subprocess.check_call([python_bin, "-m", "pytest", "tests/stats"])

# ----------------------------
# 5. Run all example files
# ----------------------------
print("\nRunning example tests...")
subprocess.check_call([python_bin, "-m", "pytest", "tests/examples"])

# ----------------------------
# 6. Run all util tests
# ----------------------------
print("\nRunning util tests...")
subprocess.check_call([python_bin, "-m", "pytest", "tests/utils"])

# ----------------------------
# 7. Run torch tests (if available)
# ----------------------------
print("\nChecking for PyTorch...")
try:
    # Check if torch is available
    result = subprocess.run(
        [python_bin, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode == 0:
        torch_version = result.stdout.strip()
        print(f"PyTorch {torch_version} detected. Running torch_stats tests...")

        # Set environment variables for torch tests
        env = os.environ.copy()
        env["NUMBA_DISABLE_JIT"] = "1"

        # Check for TEST_TORCH_DEVICE environment variable
        test_device = os.environ.get("TEST_TORCH_DEVICE", "cpu")
        if test_device != "cpu":
            print(f"Testing on device: {test_device}")
            env["TEST_TORCH_DEVICE"] = test_device

        # Run torch tests
        subprocess.check_call(
            [python_bin, "-m", "pytest", "tests/torch_stats", "-v"], env=env
        )
        print("✅ Torch tests passed!")
    else:
        print("⚠️  PyTorch not installed. Skipping torch_stats tests.")
        print("   To install: pip install torch")

except subprocess.TimeoutExpired:
    print("⚠️  Torch import timed out. Skipping torch_stats tests.")
except Exception as e:
    print(f"⚠️  Could not detect PyTorch: {e}")
    print("   Skipping torch_stats tests.")

# # ----------------------------
# # 8. Run MPI-enabled tests
# # ----------------------------
# print("\nRunning MPI-enabled tests with 4 processes...")
# subprocess.check_call([mpiexec_bin, "-n", "4", python_bin, "-m", "pytest", "tests/mpi4py"])
