#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

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
# 7. Run MPI-enabled tests
# ----------------------------
print("\nRunning MPI-enabled tests with 4 processes...")
subprocess.check_call([mpiexec_bin, "-n", "4", python_bin, "-m", "pytest", "tests/mpi4py"])
