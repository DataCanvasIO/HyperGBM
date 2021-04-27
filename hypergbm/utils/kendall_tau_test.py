# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import hashlib

from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypernets.searchers import PlaybackSearcher


def kendalltau_between_sampled_and_whole(df, target_col, sample_rate=0.2, max_trials=50, reward_metric='auc',
                                         calc_top_n_kendalltau=None,
                                         exp_sampled_params=None, exp_wholedata_params=None,
                                         random_state=9527):
    """
    Calculate Kendall's tau between models rewards with sampled and whole data
    """
    if sample_rate >= 1.0:
        df_sampled = df.copy()
    else:
        df_sampled, _ = train_test_split(df, train_size=sample_rate, random_state=random_state)
    base_params = {'log_level': 'warn',
                   'target': target_col,
                   'cv': True,
                   'reward_metric': reward_metric,
                   'collinearity_detection': False,
                   'max_trials': max_trials,
                   'early_stopping_rounds': max_trials,
                   'early_stopping_time_limit': 0, }
    if exp_sampled_params is not None:
        p1 = base_params.copy()
        p1.update(exp_sampled_params)
        exp_sampled_params = p1
    else:
        exp_sampled_params = base_params.copy()
    exp_sampled = make_experiment(df_sampled,  **exp_sampled_params)
    exp_sampled.run()
    
    playback = PlaybackSearcher(exp_sampled.hyper_model.history, top_n=max_trials,
                                optimize_direction=exp_sampled.hyper_model.searcher.optimize_direction)
    if exp_wholedata_params is not None:
        p2 = base_params.copy()
        p2.update(exp_wholedata_params)
        exp_wholedata_params = p2
    else:
        exp_wholedata_params = base_params.copy()
    exp_wholedata_params['searcher'] = playback
    exp_wholedata = make_experiment(df.copy(), **exp_wholedata_params)
    exp_wholedata.run()

    r1 = [r[0] for r in sorted([(t.reward,
                                 hashlib.md5(
                                     (';'.join([f'{v}' for v in t.space_sample.vectors])).encode('utf_8')).hexdigest())
                                for t in exp_sampled.hyper_model.history.trials], key=lambda x: x[1])]

    r2 = [r[0] for r in sorted([(t.reward,
                                 hashlib.md5(
                                     (';'.join([f'{v}' for v in t.space_sample.vectors])).encode('utf_8')).hexdigest())
                                for t in exp_wholedata.hyper_model.history.trials], key=lambda x: x[1])]

    k, p = kendalltau(r1, r2)
    print(f'kendall tau:{k}, p_value:{p}')

    if calc_top_n_kendalltau is not None and calc_top_n_kendalltau > 0:
        x_top = sorted([(a, b) for a, b in zip(r1, r2)], key=lambda x: x[0], reverse=True)[:calc_top_n_kendalltau]
        k1, p1 = kendalltau([x[0] for x in x_top], [x[1] for x in x_top])
        print(f'Top {calc_top_n_kendalltau} kendall tau:{k1}, p_value:{p1}')

    return k, p, exp_sampled, exp_wholedata,
