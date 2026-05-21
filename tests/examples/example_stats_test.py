"""Run pytests on all examples/examples_stats files."""

from . import run_example

EXAMPLES_PATH = "examples/stats_examples"


def test_association_example():
    run_example("association_example.py", examples_path=EXAMPLES_PATH)


def test_binomial_example():
    run_example("binomial_example.py", examples_path=EXAMPLES_PATH)


def test_categorical_example():
    run_example("categorical_example.py", examples_path=EXAMPLES_PATH)


def test_catmultinomial_example():
    run_example("catmultinomial_example.py", examples_path=EXAMPLES_PATH)


def test_composite_example():
    run_example("composite_example.py", examples_path=EXAMPLES_PATH)


def test_conditional_example():
    run_example("conditional_example.py", examples_path=EXAMPLES_PATH)


def test_dirichlet_example():
    run_example("dirichlet_example.py", examples_path=EXAMPLES_PATH)


def test_dmvn_mixture_example():
    run_example("dmvn_mixture_example.py", examples_path=EXAMPLES_PATH)


def test_exponential_example():
    run_example("exponential_example.py", examples_path=EXAMPLES_PATH)


def test_gamma_example():
    run_example("gamma_example.py", examples_path=EXAMPLES_PATH)


def test_gaussian_example():
    run_example("gaussian_example.py", examples_path=EXAMPLES_PATH)


def test_geometric_example():
    run_example("geometric_example.py", examples_path=EXAMPLES_PATH)


def test_gmm_example():
    run_example("gmm_example.py", examples_path=EXAMPLES_PATH)


def test_heterogeneous_mixture_example():
    run_example("heterogeneous_mixture_example.py", examples_path=EXAMPLES_PATH)


def test_hidden_association_example():
    run_example("hidden_association_example.py", examples_path=EXAMPLES_PATH)


def test_hidden_markov_example():
    run_example(
        "hidden_markov_example.py", examples_path=EXAMPLES_PATH, disable_numba_jit=False
    )


def test_hierarchical_mixture_example():
    run_example("hierarchical_mixture_example.py", examples_path=EXAMPLES_PATH)


def test_icltree_example():
    run_example("icltree_example.py", examples_path=EXAMPLES_PATH)


def test_ignored_example():
    run_example("ignored_example.py", examples_path=EXAMPLES_PATH)


def test_int_plsi_example():
    run_example(
        "int_plsi_example.py", examples_path=EXAMPLES_PATH, disable_numba_jit=False
    )


def test_int_spike_example():
    run_example("int_spike_example.py", examples_path=EXAMPLES_PATH)


def test_intmultinomial_example():
    run_example("intmultinomial_example.py", examples_path=EXAMPLES_PATH)


def test_intrange_example():
    run_example("intrange_example.py", examples_path=EXAMPLES_PATH)


def test_intsetdist_example():
    run_example("intsetdist_example.py", examples_path=EXAMPLES_PATH)


def test_jmixture_example():
    run_example("jmixture_example.py", examples_path=EXAMPLES_PATH)


# def test_lda_example():
#     run_example(
#         "lda_example.py", examples_path=EXAMPLES_PATH, disable_numba_jit=False
#     )


def test_log_gaussian_example():
    run_example("log_gaussian_example.py", examples_path=EXAMPLES_PATH)


def test_markov_chain_example():
    run_example("log_gaussian_example.py", examples_path=EXAMPLES_PATH)


def test_mixture_example():
    run_example("log_gaussian_example.py", examples_path=EXAMPLES_PATH)


def test_mvn_example():
    run_example("mvn_example.py", examples_path=EXAMPLES_PATH)


def test_optional_example():
    run_example("optional_example.py", examples_path=EXAMPLES_PATH)


def test_poisson_example():
    run_example("poisson_example.py", examples_path=EXAMPLES_PATH)


def test_semi_supervised_mixture_example():
    run_example("semi_supervised_mixture_example.py", examples_path=EXAMPLES_PATH)


def test_sequence_example():
    run_example("sequence_example.py", examples_path=EXAMPLES_PATH)


def test_set_edit_example():
    run_example("set_edit_example.py", examples_path=EXAMPLES_PATH)


def test_spearman_rho_example():
    run_example("spearman_rho_example.py", examples_path=EXAMPLES_PATH)


def test_stepset_edit_example():
    run_example("stepset_edit_example.py", examples_path=EXAMPLES_PATH)
