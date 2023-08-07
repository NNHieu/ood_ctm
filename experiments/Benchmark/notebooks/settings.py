import os
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True, dotenv=True, indicator=['pyproject.toml'])
current_file_dir = Path(__file__).parent.parent.resolve()
hydra_cfg_path = current_file_dir / "configs"
# os.chdir(root)

DATA_ROOT = Path(os.environ['TORCH_DATASETS'])