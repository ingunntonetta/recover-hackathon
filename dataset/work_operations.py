import os
import random
import typing
from pathlib import Path
from dataset.metadata import MetadataDataset
import numpy as np
import polars as pl
import torch
from dataset.base import BaseDataset, index_encode_str


class WorkOperationsDataset(BaseDataset):
    competition = "hackathon-recover-x-cogito"
    resources: typing.ClassVar[list[str]] = ["train.csv", "val.csv", "test.csv", "tickets.csv"]

    __rooms: typing.ClassVar[list[str]] = [
        "andre områder", "kjøkken", "stue", "gang", "soverom",
        "bad", "bod", "vaskerom", "wc", "kjeller", "garasje"
    ]

    def __init__(
        self,
        root: str | Path,
        metadata_dataset: typing.Optional[MetadataDataset] = None,
        split: str = "train",
        seed: int | None = None,
        np_rng: np.random.Generator | None = None,
        py_rng: random.Random | None = None,
        num_clusters: int = 388
    ):
        super().__init__(root)
        self.split = split
        self.seed = seed
        self.num_clusters = num_clusters

        self.np_rng = np_rng if np_rng is not None else np.random.default_rng(seed)
        self.py_rng = py_rng if py_rng is not None else random.Random(seed)

        self.metadata_dataset = metadata_dataset

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please download it first.")

        self.data = self._load_data()
        self._load_tickets()
        self.shuffle()

    def __getitem__(self, idx):
        row = self.data.row(idx)
        project_id = row["project_id"]

        # Metadata
        meta_row = self.metadata_dataset[project_id] if self.metadata_dataset else None
        if meta_row:
            metadata = torch.cat([
                meta_row["insurance_company_one_hot"].flatten(),
                torch.tensor([
                    meta_row["recover_office_zip_code"],
                    meta_row["damage_address_zip_code"],
                    meta_row["office_distance"],
                    meta_row["case_creation_year"],
                    meta_row["case_creation_month"]
                ], dtype=torch.float32)
            ])
        else:
            metadata = torch.zeros(19, dtype=torch.float32)  # default if metadata missing

        return {
            "X": torch.tensor(row["X"]),
            "Y": torch.tensor(row["Y"]),
            "project_id": project_id,
            "room_cluster_one_hot": torch.tensor(row["room_cluster_one_hot"]),
            "calculus": row["calculus"],
            "metadata": metadata,
        }

    def _load_data(self):
        df = pl.read_csv(os.path.join(self.root, f"{self.split}.csv"))

        # room clustering
        df = df.with_columns(
            pl.col("room").map_elements(lambda r: r.lower(), return_dtype=pl.Utf8).alias("room_cluster")
        )
        df = df.with_columns(
            pl.col("room_cluster")
            .map_elements(lambda x: index_encode_str(x, len(self.__rooms), {r: i for i, r in enumerate(self.__rooms)}), return_dtype=pl.List(pl.Float32))
            .alias("room_cluster_one_hot")
        )

        # convert X, Y to index encoded
        df = df.with_columns([
            pl.col("X").map_elements(lambda x: self._index_encode(x), return_dtype=pl.List(pl.Int8)),
            pl.col("Y").map_elements(lambda x: self._index_encode(x), return_dtype=pl.List(pl.Int8))
        ])
        return df

    def _index_encode(self, labels: list[int]) -> list[int]:
        vec = np.zeros(self.num_clusters, dtype=np.int8)
        for i in labels:
            vec[i] = 1
        return vec.tolist()

    def _load_tickets(self):
        tickets_file = os.path.join(self.root, "tickets.csv")
        if os.path.exists(tickets_file):
            df = pl.read_csv(tickets_file)
            self.tickets = df
        else:
            self.tickets = None

    def _check_exists(self):
        return all(os.path.exists(os.path.join(self.root, f)) for f in self.resources)