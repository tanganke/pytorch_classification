#!/usr/bin/env python
"""
print config to console.

usage:
    python scripts/print_config.py --config-name CONFIG_NAME
"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_classification.utils import TitledLog
from pytorch_classification.utils.logging import pprint_yaml, setup_colorlogging

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
)
def main(cfg: DictConfig) -> None:
    setup_colorlogging(force=True)
    with TitledLog("configuration", log_fn=print):
        pprint_yaml(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
