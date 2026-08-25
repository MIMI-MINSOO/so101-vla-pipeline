from gr00t.experiment.data_config import So100DualCamDataConfig


class So101MarkerH10DataConfig(So100DualCamDataConfig):
    """so100_dualcam with action_horizon=10 (matches pi0_lora_marker's action_horizon=10).

    NOTE (see ../../docs/ARCHITECTURE_COMPARISON.md): this action_horizon=10 was carried
    over from an older pi0 config (pi0_lora_marker, trained on marker_50) rather than the
    action_horizon=30 that pi0_lora_marker_100 actually used on the same marker_100 dataset
    that achieved 80%+. Retraining with a matching action_horizon=30 variant is an open
    follow-up (not yet done as of this repo's last update).
    """

    action_indices = list(range(10))
