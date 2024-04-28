#! /usr/bin/env python3
R"""
train classifier, ERM optimization.

config layout:

```yaml
data:
  train_loader  # instantiatable
  [val_loader]  # instantiatable
  [test_loader] # instantiatable

model           # instantiatable

trainer         # instantiatable
```

"""
import logging
import os
from typing import Optional

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pytorch_classification.pl_modules import ERMClassificationModule
from pytorch_classification.utils.logging import pprint_yaml, setup_colorlogging
from pytorch_classification.utils import TimeIt

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="cifar10",
)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)
    pprint_yaml(OmegaConf.to_yaml(cfg))
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    # load data
    with TimeIt("load data", log.info):
        train_loader: DataLoader = instantiate(cfg.data.train_loader) if "train_loader" in cfg.data else None
        val_loader: Optional[DataLoader] = instantiate(cfg.data.val_loader) if "val_loader" in cfg.data else None
        test_loader: Optional[DataLoader] = instantiate(cfg.data.test_loader) if "test_loader" in cfg.data else None

    # load model
    with TimeIt("load model", log.info):
        model: nn.Module = instantiate(cfg.model)
        log.info(f"{'model':=^50}")
        log.info(model)
    module = ERMClassificationModule(
        model,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.optim,
    )

    trainer: pl.Trainer = instantiate(cfg.trainer)
    if train_loader is not None:
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    if test_loader is not None:
        trainer.test(
            module,
            dataloaders=test_loader,
        )


if __name__ == "__main__":
    log.info(os.getcwd())
    main()
