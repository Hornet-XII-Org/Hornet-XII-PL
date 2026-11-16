from hornet_localization.inertial_estimator import InertialEstimator


def test_estimator_exposes_parallel_work_interface():
    estimator = InertialEstimator()

    assert callable(estimator.reset)
    assert callable(estimator.update)
