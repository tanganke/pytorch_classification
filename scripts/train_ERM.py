import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from pytorch_classification.pl_modules import ERMClassificationModule
from pytorch_classification.utils.logging import pprint_yaml, setup_colorlogging

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
    trainer: pl.Trainer = instantiate(cfg.trainer)

    train_loader: DataLoader = instantiate(cfg.data.train_loader)
    val_loader: Optional[DataLoader] = instantiate(cfg.data.val_loader) if "val_loader" in cfg.data else None

    model: nn.Module = instantiate(cfg.model)
    module = ERMClassificationModule(
        model,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.optim,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    log.info(os.getcwd())
    main()
