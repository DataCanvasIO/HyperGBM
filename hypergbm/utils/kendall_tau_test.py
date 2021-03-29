# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import hashlib

from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypernets.searchers import PlaybackSearcher


def kendall_tau_between_sampled_and_whole(df, target_col, sample_rate=0.2, max_trials=50, reward_metric='auc', random_state=9527):
    """
    Calculate Kendall's tau between models rewards with sampled and whole data
    """
    df_sampled, df_test = train_test_split(df, train_size=sample_rate, random_state=random_state)
    exp_sampled = make_experiment(df_sampled,
                                  log_level='warn',
                                  target=target_col,
                                  cv=True,
                                  reward_metric=reward_metric,
                                  collinearity_detection=False,
                                  max_trials=max_trials,
                                  early_stopping_rounds=max_trials,
                                  )
    exp_sampled.run()
    playback = PlaybackSearcher(exp_sampled.hyper_model.history, top_n=max_trials)
    exp_wholedata = make_experiment(df,
                                    searcher=playback,
                                    log_level='warn',
                                    target=target_col,
                                    cv=True,
                                    collinearity_detection=False,
                                    reward_metric=reward_metric,
                                    max_trials=max_trials,
                                    early_stopping_rounds=max_trials,
                                    )
    exp_wholedata.run()

    r1 = [r[0] for r in sorted([(t.reward,
                                 hashlib.md5(
                                     (';'.join([f'{v}' for v in t.space_sample.vectors])).encode('utf_8')).hexdigest())
                                for t in exp_sampled.hyper_model.history.history], key=lambda x: x[1])]

    r2 = [r[0] for r in sorted([(t.reward,
                                 hashlib.md5(
                                     (';'.join([f'{v}' for v in t.space_sample.vectors])).encode('utf_8')).hexdigest())
                                for t in exp_wholedata.hyper_model.history.history], key=lambda x: x[1])]

    k, p = kendalltau(r1, r2)
    print(f'kendall tau:{k}, p_value:{p}')
    return k, p, exp_sampled.hyper_model.history, exp_wholedata.hyper_model.history
