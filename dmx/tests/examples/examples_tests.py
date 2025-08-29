"""Run pytests on all py files in examples folder."""
import subprocess
import os 
from pathlib import Path
file_path = Path(__file__)
examples_path = os.path.join(file_path.parent.parent.parent, 'examples')

def test_auto_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'auto_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_detailed_estimation_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'detailed_estimation_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_htsne_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'htsne_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"