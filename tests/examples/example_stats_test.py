"""Run pytests on all examples/examples_stats files."""
import subprocess
import os 
from pathlib import Path
from dmx.utils.optsutil import get_parent_directory

file_path = get_parent_directory(__file__, 4)
examples_path = "examples/stats_examples"

def test_association_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'association_example.py')],     # Command to run the script
        capture_output=True,          # Capture stdout and stderr
        text=True                     # Return output as string (instead of bytes)
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_binomial_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'binomial_example.py')],     # Command to run the script
        capture_output=True,          # Capture stdout and stderr
        text=True                     # Return output as string (instead of bytes)
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_categorical_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'categorical_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_catmultinomial_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'catmultinomial_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_composite_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'composite_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_conditional_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'conditional_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_dirichlet_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'dirichlet_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_dmvn_mixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'dmvn_mixture_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_exponential_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'exponential_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_gamma_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'gamma_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_gaussian_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'gaussian_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_geometric_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'geometric_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_gmm_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'gmm_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_heterogeneous_mixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'heterogeneous_mixture_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_hidden_association_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'hidden_association_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_hidden_markov_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'hidden_markov_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_hierarchical_mixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'hierarchical_mixture_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_icltree_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'icltree_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_ignored_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'ignored_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_int_plsi_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'int_plsi_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_int_spike_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'int_spike_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_intmultinomial_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'intmultinomial_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_intrange_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'intrange_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_intsetdist_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'intsetdist_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_jmixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'jmixture_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


# def test_lda_example():
#     result = subprocess.run(
#         ['python', os.path.join(examples_path, 'lda_example.py')],
#         capture_output=True,          
#         text=True                    
#     )

#     # Check that the script ran successfully (exit code 0)
#     assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_log_gaussian_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'log_gaussian_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_markov_chain_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'log_gaussian_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_mixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'log_gaussian_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_mvn_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'mvn_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_optional_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'optional_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_poisson_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'poisson_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_semi_supervised_mixture_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'semi_supervised_mixture_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_sequence_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'sequence_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_set_edit_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'set_edit_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_spearman_rho_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'spearman_rho_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"


def test_stepset_edit_example():
    result = subprocess.run(
        ['python', os.path.join(examples_path, 'stepset_edit_example.py')],
        capture_output=True,          
        text=True                    
    )

    # Check that the script ran successfully (exit code 0)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
