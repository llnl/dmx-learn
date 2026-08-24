"""Spark example for fitting an LDA topic model on Wikipedia text.

This script loads a small text corpus, removes stop words, maps tokens to
integers, and trains an LDA model with Spark-based sequence estimation.

Run from the repository root with:

    spark-submit --master local[4] examples/spark_examples/wikipedia_example.py
"""

import os
import sys
import time
from typing import cast

import numpy as np
from pyspark import SparkConf, SparkContext

import dmx.utils.optsutil as ops
from dmx.stats import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    LDADistribution,
    LDAEstimator,
    initialize,
    seq_encode,
    seq_estimate,
    seq_log_density_sum,
)

data_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")


def load_wiki_data() -> tuple[list[list[str]], list[str]]:

    sword_loc = os.path.join(data_loc, "stop_words")
    sword = {""}
    for file_name in ["mallet.txt"]:
        stop_word_path = os.path.join(sword_loc, file_name)
        with open(stop_word_path, "rt", encoding="utf-8") as fin:
            sword.update(fin.read().splitlines())

    wiki_loc = os.path.join(data_loc, "wiki_example")
    files = [
        os.path.join(wiki_loc, u)
        for u in filter(lambda v: v.endswith(".txt"), os.listdir(wiki_loc))
    ]
    documents = ops.flat_map(
        lambda x: x,
        [
            list(
                map(
                    lambda u: list(filter(lambda v: v not in sword, u.split(" "))),
                    ops.text_file(f),
                )
            )
            for f in files
        ],
    )
    documents = list(filter(lambda u: len(u) > 0, documents))

    vocabulary = sorted({word for doc in documents for word in doc})

    return documents, vocabulary


def create_spark_context() -> SparkContext:
    """Create a Spark context with reduced log verbosity."""
    wiki_context = SparkContext(conf=SparkConf().setAppName("wikipedia_example"))
    wiki_context.setLogLevel("ERROR")
    return wiki_context


if __name__ == "__main__":

    sc = create_spark_context()

    num_topics = 10
    print_cnt = 10
    rng = np.random.RandomState(2)
    # out = open('/Users/username/PycharmProjects/wiki_debug.log', 'wt')
    out = sys.stdout

    data, words = load_wiki_data()

    avg_size = np.mean([len(u) for u in data])

    out.write(
        f"#words = {len(words)} / #docs = {len(data)} / avg w/doc = {avg_size:f}\n"
    )

    word_map: dict[str, int] = {}
    int_data = [ops.map_to_integers(u, word_map) for u in data]
    data_cnt = [list(ops.count_by_value(u).items()) for u in int_data]
    word_map_inv = ops.get_inv_map(word_map)

    estimator0 = IntegerCategoricalEstimator(
        min_val=0, max_val=(len(word_map) - 1), pseudo_count=0.001
    )
    estimator1 = LDAEstimator(
        [estimator0] * num_topics, keys=(None, "topics"), gamma_threshold=1.0e-8
    )

    estimator = estimator1

    # The local and Spark versions differ mainly in the parallelized data.
    spark_data_cnt = sc.parallelize(data_cnt, 4)

    imm = initialize(spark_data_cnt, estimator, rng, 0.1)

    enc_data = seq_encode(spark_data_cnt, model=imm)
    prev_model = imm

    dcnt, lob_sum = seq_log_density_sum(enc_data, imm)
    old_elob = lob_sum / dcnt

    for kk in range(300):
        t0 = time.time()
        mm = seq_estimate(enc_data, estimator, prev_estimate=prev_model)
        t1 = time.time()

        dcnt, lob_sum = seq_log_density_sum(enc_data, mm)
        elob = lob_sum / dcnt

        prev_model = mm
        out.write(
            f"Iteration {kk + 1}\tE[LoB]={elob:e}\tdelta E[LoB]={elob - old_elob:e}"
            f"\tdelta time={t1 - t0:f}\n"
        )

        old_elob = elob

        if (kk + 1) % print_cnt == 0:

            lda_model = cast(LDADistribution, mm)
            topics = [cast(IntegerCategoricalDistribution, u) for u in lda_model.topics]

            for i in np.argsort(-lda_model.alpha):
                sidx = np.argsort(-topics[i].log_p_vec)
                top_words = ", ".join(
                    [
                        f"{word_map_inv[j]} ({np.exp(topics[i].log_p_vec[j]):f})"
                        for j in sidx[:10]
                    ]
                )
                out.write(f"Topic {i} [{lda_model.alpha[i]:f}]: {top_words}\n")

        out.flush()
