from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from g05.utils.training.train_utils import init_experiment_tracker


@pytest.mark.parametrize('tracker', [None, 'none', 'NONE'])
def test_disabled_tracker_does_not_initialize_external_logging(tracker):
    accelerator = Mock()
    cfg = OmegaConf.create({'logger': {'type': tracker}})
    assert init_experiment_tracker(cfg, accelerator, Path('unused')) == 'none'
    accelerator.init_trackers.assert_not_called()
