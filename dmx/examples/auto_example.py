"""Any example of how to use automatic to select an estimator from data."""
import numpy as np
from dmx.stats import *
from dmx.utils.automatic import get_estimator

if __name__ == '__main__':
    data = [(1, None, 'a', [('a', 1), ('b', 2)]), (3, 2, 'b', [('a', 1), ('b', 2), ('c', 3)]),
            (3.1, 2, 'a', [('a', 1), ('b', 2), ('c', 3)])]
    
    # Must flag use_bstats = False for standard dmx.stats models
    est = get_estimator(data, pseudo_count=1.0e-4, use_bstats=False)
    init = initialize(data,est,np.random.RandomState(1))
    model = estimate(data, est, prev_estimate=init)

    print(str(model))
