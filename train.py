from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from model.datamodule import *
from model.model import *


class CustomTrainer(pl.Trainer):
    def __init__(self, *, checkpoint, **kwargs) -> None:
        kwargs.pop("loggers", None)  # Remove loggers

        directory = kwargs.get("default_root_dir", "runs")

        # Creating default callbacks
        ckpt_path = Path(directory) / "checkpoints"
        model_checkpoint = ModelCheckpoint(dirpath=ckpt_path, **checkpoint)

        kwargs["callbacks"] = kwargs["callbacks"] or []
        kwargs["callbacks"].append(model_checkpoint)

        # Creating loggers
        if kwargs["logger"] is False:
            super().__init__(**kwargs)
            return

        # Creating default loggers
        kwargs["logger"] = [
            TensorBoardLogger(save_dir=directory, name="tensorboard", version=""),
            CSVLogger(save_dir=directory, name="csv", version=""),
        ]

        super().__init__(**kwargs)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_argument("--run-dir", type=str, default=False, required=False)
        parser.add_argument("--exp-path", required=True)
        parser.link_arguments("exp_path", "trainer.default_root_dir")


if __name__ == "__main__":
    MyLightningCLI(
        model_class=HModel,
        datamodule_class=HDataModule,
        trainer_class=CustomTrainer,
        run=True,
        save_config_callback=None,
    )
