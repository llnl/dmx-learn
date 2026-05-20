"""Fit an LDA-style mixture model to simulated bag-of-words documents."""

# pylint: disable=duplicate-code

import sys

import numpy as np

import dmx.utils.optsutil as ops
from dmx.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    LDADistribution,
    LDAEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
    seq_estimate,
    seq_initialize,
    seq_log_density_sum,
)


def make_fake_data(
    topic_count: int, doc_count: int, snr: float, p_alpha: float, seed: int
):
    word_per_doc = 100
    num_words = 10
    random_state = np.random.RandomState(seed)

    alpha1 = p_alpha * np.ones(topic_count)
    alpha1[np.arange(topic_count) >= (topic_count / 2)] = 0.0001

    alpha2 = p_alpha * np.ones(topic_count)
    alpha2[np.arange(topic_count) < (topic_count / 2)] = 0.0001

    topic_specs = [
        {
            word: snr * random_state.rand()
            + (
                1.0
                if i * num_words / topic_count
                <= word
                < (i + 1) * num_words / topic_count
                else 0.0
            )
            for word in range(num_words)
        }
        for i in range(topic_count)
    ]
    topic_distributions = [
        CategoricalDistribution(
            {
                str(word): value / float(sum(spec.values()))
                for word, value in spec.items()
            }
        )
        for spec in topic_specs
    ]

    dist1 = LDADistribution(
        topic_distributions,
        alpha1,
        len_dist=CategoricalDistribution({word_per_doc: 1.0}),
    )
    dist2 = LDADistribution(
        topic_distributions,
        alpha2,
        len_dist=CategoricalDistribution({word_per_doc: 1.0}),
    )
    mixture_dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    sampled_documents = mixture_dist.sampler(seed=1).sample(size=doc_count)
    observed_vocabulary = sorted({word for doc in sampled_documents for word in doc})

    return sampled_documents, observed_vocabulary, mixture_dist


if __name__ == "__main__":

    num_topics = 10
    print_cnt = 10
    out = sys.stdout

    # Generate data
    documents, vocabulary, _dist = make_fake_data(num_topics, 50, 0.0001, 1, 1)

    avg_size = np.mean([len(doc) for doc in documents])

    out.write(
        f"#words = {len(vocabulary)} / #docs = {len(documents)} / "
        f"avg w/doc = {avg_size:f}\n"
    )

    data_cnt = [list(ops.count_by_value(doc).items()) for doc in documents]

    # Define the estimator
    estimator1 = LDAEstimator(
        [
            CategoricalEstimator(
                pseudo_count=0.001,
                suff_stat={word: 1.0 / len(vocabulary) for word in vocabulary},
            )
        ]
        * num_topics,
        keys=(None, None),
        gamma_threshold=0.001,
    )
    estimator = MixtureEstimator([estimator1] * 2, pseudo_count=1.0)

    # Encode Data for vectorized calls
    enc_data = seq_encode(data_cnt, estimator=estimator)
    # Vectorized initialization of model
    imm = seq_initialize(enc_data, estimator, rng=np.random.RandomState(1), p=0.10)
    prev_model = imm

    # find best fitting model
    dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
    old_elob = lob_sum / dcnt
    d_elob = np.inf
    kk = -1

    while d_elob > 1.0e-8:
        kk += 1
        mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)

        dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
        elob = lob_sum / dcnt

        prev_model = mm
        out.write(
            f"Iteration {kk + 1}\tE[LOB]={elob:e}\tdelta E[LOB]={elob - old_elob:e}\n"
        )

        old_elob = elob

        if (kk + 1) % print_cnt == 0:

            out.write(f"Weights = {','.join(map(str, mm.w))}\n")
            out.write(f"Alpha_2 = {','.join(map(str, mm.components[0].alpha))}\n")
            out.write(f"Alpha_1 = {','.join(map(str, mm.components[1].alpha))}\n")
            topic_models = mm.components[0].topics

            for i in range(num_topics):
                log_prob_vec = np.asarray(list(topic_models[i].pmap.values()))
                vals = np.asarray(list(topic_models[i].pmap.keys()))

                sidx = np.argsort(-log_prob_vec)
                top_words = ", ".join(
                    [f"{vals[j]} ({np.exp(log_prob_vec[j]):f})" for j in sidx]
                )
                out.write(f"Topic {i}: {top_words}\n")

        out.flush()
