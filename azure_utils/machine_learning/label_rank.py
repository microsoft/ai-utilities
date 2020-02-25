"""
ai-utilities - machine_learning/label_rank.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import numpy as np
import pandas as pd


def score_rank(scores):
    """
    Add Rank to Series

    :param scores: Series to Rank
    :return: Ranked Series
    """
    return pd.Series(scores).rank(ascending=False)


def label_index(label, label_order):
    """
    Label Index

    :param label: Label to apply to items
    :param label_order: Order of labels to apply
    :return: Labeled Item
    """
    loc = np.where(label == label_order)[0]
    if loc.shape[0] == 0:
        return None
    return loc[0]


def label_rank(label, scores, label_order) -> int:
    """
    Add Labels based on Rank

    :param label: Label to assign to item
    :param scores: Score to rank item
    :param label_order: Order of Labels
    :return: Return Index
    """
    loc = label_index(label, label_order)
    if loc is None:
        return len(scores) + 1
    return score_rank(scores)[loc]
