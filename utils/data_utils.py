import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen, mode: str = "train", sample_size: int = -1):
        super().__init__()
        if mode not in {"train", "valid", "test"}:
            raise ValueError("mode must be one of: train, valid, test")

        self.dataset = dataset
        self.maxlen = maxlen
        self.mode = mode
        self.trainData, self.valData, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

        dataset_name = os.path.basename(os.path.normpath(self.dataset))
        interaction_path = os.path.join(self.dataset, f"{dataset_name}.txt")
        with open(interaction_path, "r") as f:
            for line in f:
                user, *raw_items = line.strip().split(" ")
                user = int(user) - 1
                items = [int(item) for item in raw_items]
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

                if len(items) >= 3:
                    train_items = items[:-2]
                    length = min(len(train_items), self.maxlen)
                    for t in range(length):
                        self.trainData.append(
                            [train_items[:-length + t], train_items[-length + t]]
                        )
                    self.valData[user] = [items[:-2], items[-2]]
                    self.testData[user] = [items[:-1], items[-1]]
                else:
                    for t in range(len(items)):
                        self.trainData.append([items[:-len(items) + t], items[-len(items) + t]])
                    self.valData[user] = []
                    self.testData[user] = []

        attr_path = os.path.join(self.dataset, f"{dataset_name}_item2attributes.json")
        self.item2attribute_ids = json.load(open(attr_path, "r"))
        if self.item2attribute_ids and isinstance(next(iter(self.item2attribute_ids.keys())), str):
            self.item2attribute_ids = {int(k): v for k, v in self.item2attribute_ids.items()}
            self.item2attribute_ids = dict(sorted(self.item2attribute_ids.items()))

        if min(self.item2attribute_ids.keys()) != 1:
            raise ValueError("item ids in item2attribute mapping must start from 1")
        if max(self.item2attribute_ids.keys()) != self.m_item:
            raise ValueError("item ids in item2attribute mapping must match the dataset item range")

        self.n_user += 1
        self.m_item += 1

        self.allPos = None
        if mode == "train":
            sample_path = os.path.join(self.dataset, f"{dataset_name}_sample.txt")
            self.allPos = {}
            with open(sample_path, "r") as f:
                for line in f:
                    user, *raw_items = line.strip().split(" ")
                    self.allPos[int(user) - 1] = [int(item) for item in raw_items]
            self.trainData = self.sample_data(self.trainData, sample_size)
        elif mode == "valid":
            self.valData = [
                [self.valData[user][0], self.valData[user][1]]
                for user in self.valData
                if len(self.valData[user]) > 0
            ]
            self.valData = self.sample_data(self.valData, sample_size)
        else:
            self.testData = [
                [self.testData[user][0], self.testData[user][1]]
                for user in self.testData
                if len(self.testData[user]) > 0
            ]
            self.testData = self.sample_data(self.testData, sample_size)

    @staticmethod
    def sample_data(collection: list, sample_size: int):
        if sample_size <= 0 or len(collection) <= sample_size:
            return collection
        sampled_indices = np.random.choice(len(collection), size=sample_size, replace=False)
        return [collection[i] for i in sampled_indices]

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.trainData[idx]
        if self.mode == "valid":
            return self.valData[idx]
        return self.testData[idx]

    def __len__(self):
        if self.mode == "train":
            return len(self.trainData)
        if self.mode == "valid":
            return len(self.valData)
        return len(self.testData)


@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        max_len = max(max(len(seq) for seq in seqs), 2)

        inputs, inputs_mask = [], []
        for seq in seqs:
            pad_len = max_len - len(seq)
            inputs.append([0] * pad_len + seq)
            inputs_mask.append([0] * pad_len + [1] * len(seq))

        return {
            "inputs": torch.LongTensor(inputs),
            "inputs_mask": torch.FloatTensor(inputs_mask),
            "labels": torch.LongTensor(labels),
        }
