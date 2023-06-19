from lightning import Trainer
from lightning.pytorch.cli import LightningCLI

from model.model import *
from model.datamodule import * 
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


# TODO : Look micro vs macro F1 etc : source https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
# TODO : Add more metrics (precision, recall, AUC, ROC, etc) for both micro & micro
# TODO : Add those metrics for each level
# TODO : Construct a custom metric that takes into account the hiearchy ( mean of f1 across levels?)

# TODO : Add precision-recall plots / AUC plot / confusion matrix plot etc
# TODO : Add stage test 
# TODO : verify that test and train labels match

# TODO : https://towardsdatascience.com/https-medium-com-noa-weiss-the-hitchhikers-guide-to-hierarchical-classification-f8428ea1e076
# TODO : https://towardsdatascience.com/hierarchical-performance-metrics-and-where-to-find-them-7090aaa07183
# TODO : Use more advanced model 
# TODO : N-binary implementation 
# TODO : Coherent Implementation (CHMC) see github
# TODO : Personal implementation of TOP down approach

class CustomTrainer(pl.Trainer):
    def __init__(self, *, checkpoint, **kwargs) -> None:

        # # ! this is hard coded here, find a better way to do this 
        # self.tree = create_hiarchy_tree(categories_path="data/category_parent.csv")
        # self.label_encoders = {}
        # self.train_df = pd.read_csv("/home/alaassairi/internship/HC/data/data_train.csv")

        # for level in range(self.tree.depth()+1) :
        #     # Collapsing level
        #     console.print(f"Collapsing categories - level : {level}")
        #     self.train_df[f"category_id_level_{level}"] = self.train_df[f"category_id"].apply(lambda i : collapse_children(self.tree, i, level)  )
            
        #     # Encoding the categories to match [0, n_classes-1]
        #     self.label_encoders[level]=preprocessing.LabelEncoder()
        #     self.label_encoders[level].fit(self.train_df[f"category_id_level_{level}"].values)


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

if __name__ == '__main__':
    MyLightningCLI(
        model_class=HModel,
        datamodule_class=HDataModule,
        trainer_class=CustomTrainer,
        run=True,
        save_config_callback=None,

    )