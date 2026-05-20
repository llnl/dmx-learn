"""Fit a document mixture model over author sets and title word sequences."""

# pylint: disable=duplicate-code

import json
import os

import numpy as np

from dmx.stats import (
    BernoulliSetEstimator,
    CategoricalEstimator,
    CompositeEstimator,
    MixtureEstimator,
    SequenceEstimator,
)
from dmx.utils.estimation import optimize
from dmx.utils.optsutil import get_parent_directory

if __name__ == "__main__":

    # Load the NIPs data
    path_to_data = os.path.join(
        get_parent_directory(__file__, 4), "data", "nips", "all_submissions.json"
    )
    with open(path_to_data, "rt", encoding="utf-8") as fin:
        data = json.load(fin)

    # Extract the author sets
    papers = [
        ([v["id"] for v in u["authors"]], [v.lower() for v in u["title"]]) for u in data
    ]
    authors = {author for paper in papers for author in paper[0]}
    words = {word for paper in papers for word in paper[1]}

    est1 = BernoulliSetEstimator(
        pseudo_count=1.0e-8, suff_stat={u: 0.5 for u in authors}
    )
    est2 = SequenceEstimator(
        CategoricalEstimator(
            pseudo_count=1.0, suff_stat={w: 1.0 / len(words) for w in words}
        )
    )
    est3 = CompositeEstimator((est1, est2))
    est = MixtureEstimator([est3] * 10)

    model = optimize(papers, est, init_p=0.10, rng=np.random.RandomState(1), max_its=10)

    for comp in model.components:
        print(sorted(comp.dists[0].pmap.items(), key=lambda u: u[1], reverse=True)[:10])
        print(
            sorted(comp.dists[1].dist.pmap.items(), key=lambda u: u[1], reverse=True)[
                :10
            ]
        )
