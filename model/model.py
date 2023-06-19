import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    FBetaScore,
    Precision,
    Recall,
)

from model.common import *
from utils.rich import *

logger = TensorBoardLogger("tb_logs", name="HModel")


class HModel(pl.LightningModule):
    def __init__(
        self,
        architecture: str = "simple",
        strategy: str = "BIGBANG",  # TODO : change those to literals
        num_target_classes_by_level: dict = {0: 101, 1: 86, 2: 58, 3: 22},
        input_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 10,
        threshold: float = 0.5,
        level_weights: dict = {0: 1, 1: 2, 2: 3, 3: 4},
    ):
        # * Initialization
        super(HModel, self).__init__()
        self.architecture = architecture
        self.strategy = strategy
        self.num_target_classes_by_level = num_target_classes_by_level
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.threshold = threshold
        self.level_weights = level_weights

        # * Metrics for Traditional (Flat) Multi class classification :
        # ! pay attention to the required input of each metric? logits vs soft vs ..
        self.metrics = {}

        for level in range(len(self.num_target_classes_by_level)):
            self.metrics[level] = self.generate_metrics_for_level(level)
        console.print("[green] Finished generating metrics")

        # * Model Architecture

        if self.architecture == "MLP":
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 768),
                torch.nn.ReLU(),
                torch.nn.Linear(768, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 101),  # ! num_target_classes_by_level[0]
                torch.nn.ReLU(),
            )
        elif self.architecture == "RESNET":
            self.model = torch.hub.load(
                "pytorch/vision", "resnet50", weights="IMAGENET1K_V2"
            )

        # elif self.architecture =="RESNET":
        #     self.model = SAINT(
        #         categories = tuple(cat_dims),
        #         num_continuous = self.input_size,
        #         dim = opt.embedding_size,
        #         dim_out = 1,
        #         depth = opt.transformer_depth,
        #         heads = opt.attention_heads,
        #         attn_dropout = opt.attention_dropout,
        #         ff_dropout = opt.ff_dropout,
        #         mlp_hidden_mults = (4, 2),
        #         cont_embeddings = opt.cont_embeddings,
        #         attentiontype = opt.attentiontype,
        #         final_mlp_style = opt.final_mlp_style,
        #         y_dim = y_dim
        #         )

        self.softmax = torch.nn.Softmax(dim=1)

        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        # Classes Hierarchy Adjucency Matrices

        # Load tensor A from disk
        self.A = torch.load("A.pt").type(torch.float16).to("cuda")

        # Load tensor B from disk
        self.B = torch.load("B.pt").type(torch.float16).to("cuda")

        # Load tensor C from disk
        self.C = torch.load("C.pt").type(torch.float16).to("cuda")

        # * Saving hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        # Model Compuatation Code
        out = self.model(x)
        # out = self.softmax(out)
        return out

    def convert_labels_old(self, preds, A):
        preds = preds.type(torch.float16)  # ! attention here
        A = A.type(torch.float16)
        result = torch.matmul(preds, A)
        indices = torch.nonzero(result == 1, as_tuple=False)[:, 1]
        return indices

    def step(self, batch):
        x, y = batch

        # level 0 : f1_score, accuracy*1

        # level 1 : f1_score, accuracy*7

        # level 3 : f1_score, accurcy *10

        # HAccurcy = weighted_mean(accuracy)/n_levels

        # Model N for level N

        # shape of y = (B, NC, 4 ) : (Batch_size, Number of Classes, Depth of hierarchy tree)
        # shape of x = (B, NF )   : (Batch_size, Number of Features)

        y_hat = self.forward(x)  # shape of y_hat = (B, NC)

        y_hat_soft = self.softmax(y_hat)
        y_hat_hard = torch.zeros_like(y_hat_soft)
        max_indices = torch.argmax(y_hat_soft, 1)
        y_hat_hard.scatter_(1, max_indices.unsqueeze(1), 1)

        y_hat_1 = self.convert_labels_old(y_hat_hard, self.A)
        y_hat_2 = self.convert_labels_old(y_hat_hard, torch.matmul(self.A, self.B))
        y_hat_3 = self.convert_labels_old(
            y_hat_hard, torch.matmul(torch.matmul(self.A, self.B), self.C)
        )

        y_0 = y[:, 0]  # level 0 (flat categories)
        y_1 = y[:, 1]  # level 1
        y_2 = y[:, 2]  # level 2
        y_3 = y[:, 3]  # level 3

        return y_0, y_hat, y_1, y_hat_1, y_2, y_hat_2, y_3, y_hat_3

    def training_step(self, batch, batch_idx):
        y, y_hat, y_1, y_hat_1, y_2, y_hat_2, y_3, y_hat_3 = self.step(batch)

        loss = F.cross_entropy(y_hat, y)  # level 0 loss

        if self.strategy == "BIGBANG ":
            loss_1 = self.kl_loss(
                y_hat_1.type(torch.float), y_1.type(torch.float)
            )  # level 1 loss
            loss_2 = self.kl_loss(
                y_hat_2.type(torch.float), y_2.type(torch.float)
            )  # level 2 loss
            loss_3 = self.kl_loss(
                y_hat_3.type(torch.float), y_3.type(torch.float)
            )  # level 3 loss

            loss = loss + 2 * loss_1 + 3 * loss_2 + 4 * loss_3

        y_hat_soft = self.softmax(y_hat)
        self.log("Train_loss", loss, prog_bar=True, sync_dist=True)

        self.train_output_list.append(
            {
                "y": y,
                "y_hat": y_hat_soft,
                "y_1": y_1,
                "y_hat_1": y_hat_1,
                "y_2": y_2,
                "y_hat_2": y_hat_2,
                "y_3": y_3,
                "y_hat_3": y_hat_3,
            }
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        y, y_hat, y_1, y_hat_1, y_2, y_hat_2, y_3, y_hat_3 = self.step(batch)

        val_loss = F.cross_entropy(y_hat, y)  # level 0 loss

        if self.strategy == "BIGBANG ":
            loss_1 = self.kl_loss(
                y_hat_1.type(torch.float), y_1.type(torch.float)
            )  # level 1 loss
            loss_2 = self.kl_loss(
                y_hat_2.type(torch.float), y_2.type(torch.float)
            )  # level 2 loss
            loss_3 = self.kl_loss(
                y_hat_3.type(torch.float), y_3.type(torch.float)
            )  # level 3 loss

            val_loss = val_loss + 2 * loss_1 + 3 * loss_2 + 4 * loss_3

        y_hat_soft = self.softmax(y_hat)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.val_output_list.append(
            {
                "y": y,
                "y_hat": y_hat_soft,
                "y_1": y_1,
                "y_hat_1": y_hat_1,
                "y_2": y_2,
                "y_hat_2": y_hat_2,
                "y_3": y_3,
                "y_hat_3": y_hat_3,
            }
        )
        return {"val_loss": val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=self.learning_rate,
            epochs=self.max_epochs,
            optimizer=optimizer,
            steps_per_epoch=100,  # ! self.num_training_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )

        self.val_output_list = []
        self.train_output_list = []
        return {"optimizer": optimizer, "scheduler": scheduler}

    def on_train_epoch_end(self):
        preds = []
        targets = []

        preds_1 = []
        targets_1 = []

        preds_2 = []
        targets_2 = []

        preds_3 = []
        targets_3 = []
        for output in self.train_output_list:
            preds.append(output["y_hat"])
            targets.append(output["y"])

            preds_1.append(output["y_hat_1"])
            targets_1.append(output["y_1"])

            preds_2.append(output["y_hat_2"])
            targets_2.append(output["y_2"])

            preds_3.append(output["y_hat_3"])
            targets_3.append(output["y_3"])

        targets = torch.cat(targets)
        preds = torch.cat(preds)

        targets_1 = torch.cat(targets_1)
        preds_1 = torch.cat(preds_1)

        targets_2 = torch.cat(targets_2)
        preds_2 = torch.cat(preds_2)

        targets_3 = torch.cat(targets_3)
        preds_3 = torch.cat(preds_3)

        results = self.calculate_metrics_big_bang(
            targets,
            preds,
            targets_1,
            preds_1,
            targets_2,
            preds_2,
            targets_3,
            preds_3,
            "train",
        )

        self.train_output_list.clear()

        return results

    def on_validation_epoch_end(self):
        preds = []
        targets = []

        preds_1 = []
        targets_1 = []

        preds_2 = []
        targets_2 = []

        preds_3 = []
        targets_3 = []

        for output in self.val_output_list:
            preds.append(output["y_hat"])
            targets.append(output["y"])

            preds_1.append(output["y_hat_1"])
            targets_1.append(output["y_1"])

            preds_2.append(output["y_hat_2"])
            targets_2.append(output["y_2"])

            preds_3.append(output["y_hat_3"])
            targets_3.append(output["y_3"])

        targets = torch.cat(targets)
        preds = torch.cat(preds)

        targets_1 = torch.cat(targets_1)
        preds_1 = torch.cat(preds_1)

        targets_2 = torch.cat(targets_2)
        preds_2 = torch.cat(preds_2)

        targets_3 = torch.cat(targets_3)
        preds_3 = torch.cat(preds_3)

        results = self.calculate_metrics_big_bang(
            targets,
            preds,
            targets_1,
            preds_1,
            targets_2,
            preds_2,
            targets_3,
            preds_3,
            "val",
        )

        self.val_output_list.clear()

        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def generate_metrics_for_level(self, level):
        num_classes = self.num_target_classes_by_level[level]
        metrics = {
            # f"auroc_macro_{level}": AUROC(task="multiclass", num_classes=num_classes, average="macro").to("cuda"),
            # f"ap_macro_{level}": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro").to("cuda"),
            f"accuracy_macro_{level}": Accuracy(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="macro",
            ).to("cuda"),
            f"accuracy_micro_{level}": Accuracy(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="micro",
            ).to("cuda"),
            f"precision_macro_{level}": Precision(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="macro",
            ).to("cuda"),
            f"precision_micro_{level}": Precision(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="micro",
            ).to("cuda"),
            f"recall_macro_{level}": Recall(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="macro",
            ).to("cuda"),
            f"recall_micro_{level}": Recall(
                task="multiclass",
                threshold=self.threshold,
                num_classes=num_classes,
                average="micro",
            ).to("cuda"),
            f"f1_macro_{level}": FBetaScore(
                task="multiclass",
                threshold=self.threshold,
                beta=1.0,
                num_classes=num_classes,
                average="macro",
            ).to("cuda"),
            f"f1_micro_{level}": FBetaScore(
                task="multiclass",
                threshold=self.threshold,
                beta=1.0,
                num_classes=num_classes,
                average="micro",
            ).to("cuda"),
        }
        return metrics

    def calculate_metrics_big_bang(
        self,
        targets,
        preds,
        targets_1,
        preds_1,
        targets_2,
        preds_2,
        targets_3,
        preds_3,
        mode="",
    ):
        # console.print(self.trainer.tree)
        results = {}
        for level, metrics in self.metrics.items():
            if level == 0:
                # Flat classification metrics for level 0
                for metric_name, metric in metrics.items():
                    results[f"{metric_name}_{mode}"] = metric(preds, targets)

            elif level == 1:
                # targets and preds should be collapsed to level "i" before calculating metrics
                for metric_name, metric in metrics.items():
                    results[f"{metric_name}_{mode}"] = metric(preds_1, targets_1)

            elif level == 2:
                for metric_name, metric in metrics.items():
                    results[f"{metric_name}_{mode}"] = metric(preds_2, targets_2)

            elif level == 3:
                for metric_name, metric in metrics.items():
                    results[f"{metric_name}_{mode}"] = metric(preds_3, targets_3)

        # * Custom Metrics For Hierarchical Multi-class classification
        # Here we calculate the weighted metric per level
        metric_names = [
            "accuracy_macro",
            "accuracy_micro",
            "precision_macro",
            "precision_micro",
            "recall_macro",
            "recall_macro",
            "recall_micro",
            "f1_macro",
            "f1_micro",
        ]

        for metric_name in metric_names:
            sum_weights = sum([i for i in self.level_weights.values()])
            results[f"H_{metric_name}_{mode}"] = (
                self.level_weights[3] * results[f"{metric_name}_3_{mode}"]
                + self.level_weights[2] * results[f"{metric_name}_2_{mode}"]
                + self.level_weights[1] * results[f"{metric_name}_1_{mode}"]
                + self.level_weights[0] * results[f"{metric_name}_0_{mode}"]
            ) / sum_weights

        # logging the results
        for key, value in results.items():
            if "H_f1" in key or "H_accuracy" in key:
                prog_par = True
                syn_dist = True
            else:
                prog_par = False
                syn_dist = False
            self.log(key, value, prog_bar=prog_par, sync_dist=syn_dist)

        return results

    # ! Fix the num of training steps here
    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     if self.trainer.max_steps != -1:
    #         return self.trainer.max_steps

    #     limit_batches = self.trainer.limit_train_batches
    #     batches = len(self.trainer.train_dataloader())
    #     batches = (min(batches, limit_batches) if isinstance(
    #         limit_batches, int) else int(limit_batches * batches))

    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)

    #     effective_accum = self.trainer.accumulate_grad_batches * num_devices
    #     return (batches // effective_accum) * self.trainer.max_epochs
