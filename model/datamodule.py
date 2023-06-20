import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from pyparsing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from model.common import *
from model.dataset import *


class HDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_src_path,
        test_src_path,
        categories_path,
        batch_size: int = 10,
        nworkers: int = 8,
        val_size: float = 0.2,
        seed: int = 42,
        verbose: bool = True,
    ):
        super().__init__()
        self.label_enconders = {}

        self.train_src_path = train_src_path
        self.test_src_path = test_src_path

        self.nworkers = nworkers
        self.verbose = verbose
        self.seed = seed
        self.val_size = val_size
        self.common_params = dict(
            cache_name=None, nworkers=self.nworkers, verbose=self.verbose
        )

        self.batch_size = batch_size
        self.train_df = pd.read_csv(self.train_src_path)
        self.test_df = pd.read_csv(self.test_src_path)

        self.tree = create_hiarchy_tree(
            categories_path=categories_path
        )  # be careful of label encoding here

    def _prepare_dataset(
        self,
        df,
        target_columns,
    ):
        return HDataset(df, target_columns)

    def setup(self, stage: str):
        # This runs across all GPUs and it is safe to make state assignments here

        # # stages "fit", "predict", "test"

        if stage == "fit":
            console.rule("Stage Fit")

            target_columns = []
            # Expanding the categories by level
            for level in range(
                self.tree.depth() + 1
            ):  # TODO: change this to tree.depth()
                # Collapsing level
                console.print(f"Collapsing categories - level : {level}")
                self.train_df[f"category_id_level_{level}"] = self.train_df[
                    f"category_id"
                ].apply(lambda i: collapse_children(self.tree, i, level))

                # Encoding the categories to match [0, n_classes-1]
                self.label_enconders[level] = preprocessing.LabelEncoder()
                self.label_enconders[level].fit(
                    self.train_df[f"category_id_level_{level}"].values
                )

                self.train_df[f"category_id_level_{level}"] = self.label_enconders[
                    level
                ].transform((self.train_df[f"category_id_level_{level}"].values))

                def create_adjacency_matrix(
                    labels_level_i, labels_level_i_plus, level_i, label_enconders, tree
                ):
                    labels_i = sorted(list(set(labels_level_i)))
                    labels_i_plus = sorted(list(set(labels_level_i_plus)))
                    adjacency_matrix = np.zeros((len(labels_i), len(labels_i_plus)))

                    for i in labels_i:
                        for i_plus in labels_i_plus:
                            i_decoded = label_enconders[level_i].inverse_transform([i])[
                                0
                            ]
                            i_plus_decoded = label_enconders[
                                level_i + 1
                            ].inverse_transform([i_plus])[0]
                            parent_label = tree.parent(i_decoded).identifier
                            if parent_label == i_plus_decoded:
                                adjacency_matrix[i, i_plus] = 1

                    return adjacency_matrix

                target_columns.append(f"category_id_level_{level}")

            # TODO Remove this hardcoded block from here after finishing
            A = create_adjacency_matrix(
                self.train_df[f"category_id_level_{0}"].values,
                self.train_df[f"category_id_level_{1}"],
                level_i=0,
                label_enconders=self.label_enconders,
                tree=self.tree,
            )
            B = create_adjacency_matrix(
                self.train_df[f"category_id_level_{1}"].values,
                self.train_df[f"category_id_level_{2}"],
                level_i=1,
                label_enconders=self.label_enconders,
                tree=self.tree,
            )
            C = create_adjacency_matrix(
                self.train_df[f"category_id_level_{2}"].values,
                self.train_df[f"category_id_level_{3}"],
                level_i=2,
                label_enconders=self.label_enconders,
                tree=self.tree,
            )

            self.A = torch.tensor(A, dtype=torch.long)
            self.B = torch.tensor(B, dtype=torch.long)
            self.C = torch.tensor(C, dtype=torch.long)

            # save tensor A to disk
            torch.save(self.A, "A.pt")

            # Save tensor B to disk
            torch.save(self.B, "B.pt")

            # Save tensor C to disk
            torch.save(self.C, "C.pt")

            # A (encoded) : 101 -> B (encoded) : 86     From level 0 to level 1
            # B (encoded ): 86  -> C (encoded) : 58     From level 1 to level 2
            # C (encoded ): 69  -> D (encoded) : 22     From level 2 to level 3

            train_split_df, val_split_df, _, _ = train_test_split(
                self.train_df,
                self.train_df.category_id,
                test_size=self.val_size,
                random_state=self.seed,
            )

            train_split_df.drop(inplace=True, columns=["category_id"])
            val_split_df.drop(inplace=True, columns=["category_id"])

            console.print("Preparing Training Dataset")
            self.train_ds: HDataset = self._prepare_dataset(
                train_split_df, target_columns
            )
            console.print("Preparing Validation Dataset")
            self.val_ds: HDataset = self._prepare_dataset(val_split_df, target_columns)
            console.print("[green]Finished Loading Datasets")

        if stage == "test":
            console.rule("Stage Test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds,  # TODO change this to test_ds
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            shuffle=False,
            pin_memory=True,
        )
