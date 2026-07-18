import sys

from run_lora_dpzero_exploration import DEFAULT_CONFIG, DEFAULT_SELECTION
from run_lora_dpzero_stage import main


if __name__ == "__main__":
    defaults = [
        "--runner-module", "run_lora_dpzero_exploration",
        "--config", str(DEFAULT_CONFIG),
        "--selection", str(DEFAULT_SELECTION),
    ]
    sys.argv[1:1] = defaults
    main()
