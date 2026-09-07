import pytest

from experiments.libero.eval_libero_parallel import _resolve_num_parallel


@pytest.mark.parametrize('workers,trials,expected', [(10,1,1), (10,50,10), (4,3,3)])
def test_parallelism_stays_within_trial_budget(workers, trials, expected):
    assert _resolve_num_parallel(workers, trials) == expected


@pytest.mark.parametrize('workers,trials', [(0,1), (1,0), (-1,10), (10,-1)])
def test_invalid_trial_budget_is_rejected(workers, trials):
    with pytest.raises(ValueError):
        _resolve_num_parallel(workers, trials)
