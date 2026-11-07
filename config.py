__version__ = "v6"

from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class PEConfig:
    T: int = 10                        # number of iterations
    N_syn_per_class: int = 5        # number of generated samples per class
    T_param: int = 5
    sigma: float = 1.0  #T_param * math.sqrt(2)     # noise multiplier
    H: float = 2 * T_param * 0.1                    # threshold for DP Nearest Neighbors Histogram
    lookahead_k: int = 8
    delta = 1e-5

    # VARIATION_API/SDEdit noise schedule
    # transfer_strength is AudioLDM's 0-1 version of noise timesteps (0 = no noise, 1 = full noise)
    transfer_strength_start: float = 0.98
    transfer_strength_end: float = 0.60

    urbansound8k_root: str = "/path/to/sound_datasets/urbansound8k"
    classes: tuple = (
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
    )
    test_classes: Optional[tuple] = ("dog_bark",)   # set to None to run every class

    max_private_samples_per_class: int = 10     # set to None to use full class (800-1000 samples each)
    subsample_seed: int = 0

    audioldm_model_name: str = "audioldm-s-full-v2"
    # AudioLDM's CLI requires duration to be a multiple of 2.5s
    # UrbanSound8K clips are <=4s, so 5.0 is the smallest valid multiple of 2.5 that's still >= 4 and avoids truncating
    audioldm_duration_s: float = 5.0
    audioldm_ddim_steps: int = 50
    # No batch-size because random_api/variation_api generate exactly one sample per audioldm call

    output_dir: str = "/path/to/output"
