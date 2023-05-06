from hypernets.core.objective import Objective


class FeatureUsageObjective(Objective):
    def __init__(self):
        super(FeatureUsageObjective, self).__init__('feature_usage', 'min', need_train_data=False,
                                                    need_val_data=True, need_test_data=False)

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        return estimator.data_pipeline[0].features[0][1].steps[0][1].feature_usage()

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        return estimator.data_pipeline[0].features[0][1].steps[0][1].feature_usage()

