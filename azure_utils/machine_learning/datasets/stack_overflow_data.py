"""
AI-Utilities - stack_overflow_data.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

import pandas as pd

from azure_utils.utilities import (
    clean_text,
    random_merge,
    read_csv_gz,
    round_sample_strat,
)


def split_duplicates(dupes, label_column, match, questions, show_output, test_size):
    """

    :param dupes:
    :param label_column:
    :param match:
    :param questions:
    :param show_output:
    :param test_size:
    :return:
    """
    dupes_test = round_sample_strat(dupes, dupes[label_column], frac=test_size)
    dupes_train = dupes[~dupes.Id.isin(dupes_test.Id)]
    assert (
        dupes_test[label_column].unique().shape[0]
        == dupes[label_column].unique().shape[0]
    )
    # The relevant columns for text pairs data.
    balanced_pairs_columns = [
        "Id_x",
        "AnswerId_x",
        "Text_x",
        "Id_y",
        "Text_y",
        "AnswerId_y",
        "Label",
        "n",
    ]
    # Use AnswerId to pair each training dupe with its matching question and also with N-1 questions not its match.
    balanced_pairs_train = random_merge(dupes_train, questions, number_to_merge=match)
    # Label records by matching AnswerIds.
    # noinspection PyUnresolvedReferences
    balanced_pairs_train["Label"] = (
        balanced_pairs_train.AnswerId_x == balanced_pairs_train.AnswerId_y
    ).astype(int)
    # Keep only the relevant data.
    balanced_pairs_train = balanced_pairs_train[balanced_pairs_columns]
    # Sort the data by dupe ID and Label.
    balanced_pairs_train.sort_values(
        by=["Id_x", "Label"], ascending=[True, False], inplace=True
    )
    # Use AnswerId to pair each testing dupe with all questions.
    balanced_pairs_test = random_merge(
        dupes_test, questions, number_to_merge=questions.shape[0]
    )
    # Label records by matching AnswerIds.
    # noinspection PyUnresolvedReferences
    balanced_pairs_test["Label"] = (
        balanced_pairs_test.AnswerId_x == balanced_pairs_test.AnswerId_y
    ).astype(int)
    # Keep only the relevant data.
    balanced_pairs_test = balanced_pairs_test[balanced_pairs_columns]
    # Sort the data by dupe ID and Label.
    balanced_pairs_test.sort_values(
        by=["Id_x", "Label"], ascending=[True, False], inplace=True
    )
    # Report on the datasets.
    if show_output:
        print(
            "balanced_pairs_train: {:,} rows with {:.2%} matches".format(
                balanced_pairs_train.shape[0], balanced_pairs_train.Label.mean()
            )
        )
        print(
            "balanced_pairs_test: {:,} rows with {:.2%} matches".format(
                balanced_pairs_test.shape[0], balanced_pairs_test.Label.mean()
            )
        )
    return balanced_pairs_test, balanced_pairs_train, dupes_test


def clean_data(answers, dupes, min_dupes, min_text, questions, show_output):
    """

    :param answers:
    :param dupes:
    :param min_dupes:
    :param min_text:
    :param questions:
    :param show_output:
    :return:
    """
    for dataframe in (questions, dupes, answers):
        dataframe["Text"] = dataframe.Text0.apply(clean_text).str.lower()
    questions = questions[questions.Text.str.len() > 0]
    answers = answers[answers.Text.str.len() > 0]
    dupes = dupes[dupes.Text.str.len() > 0]
    if show_output:
        print(questions.iloc[0, 1])
        print(questions.iloc[0, 3])
    # First, remove dupes that are questions, then remove duplicated questions and dupes.
    dupes = dupes[~dupes.index.isin(questions.index)]
    questions = questions[~questions.index.duplicated(keep="first")]
    dupes = dupes[~dupes.index.duplicated(keep="first")]
    # Keep only questions with answers and dupes, answers to questions, and dupes of questions.
    questions = questions[
        questions.AnswerId.isin(answers.index) & questions.AnswerId.isin(dupes.AnswerId)
    ]
    answers = answers[answers.index.isin(questions.AnswerId)]
    dupes = dupes[dupes.AnswerId.isin(questions.AnswerId)]
    verify_data_integrity(answers, dupes, questions)
    # Report on the data.
    if show_output:
        print("Text statistics:")
        print(
            pd.DataFrame(
                [
                    questions.Text.str.len().describe().rename("questions"),
                    answers.Text.str.len().describe().rename("answers"),
                    dupes.Text.str.len().describe().rename("dupes"),
                ]
            )
        )
        print("\nDuplication statistics:")
        print(
            pd.DataFrame(
                [dupes.AnswerId.value_counts().describe().rename("duplications")]
            )
        )
        print(
            "\nLargest class: {:.2%}".format(
                dupes.AnswerId.value_counts().max() / dupes.shape[0]
            )
        )
    # Reset each dataframe's index.
    questions.reset_index(inplace=True)
    answers.reset_index(inplace=True)
    dupes.reset_index(inplace=True)
    # Apply the minimum text length to questions and dupes.
    questions = questions[questions.Text.str.len() >= min_text]
    dupes = dupes[dupes.Text.str.len() >= min_text]
    # Keep only questions with dupes, and dupes of questions.
    label_column = "AnswerId"
    questions = questions[questions[label_column].isin(dupes[label_column])]
    dupes = dupes[dupes[label_column].isin(questions[label_column])]
    # Restrict the questions to those with a minimum number of dupes.
    answerid_count = dupes.groupby(label_column)[label_column].count()
    answerid_min = answerid_count.index[answerid_count >= min_dupes]
    questions = questions[questions[label_column].isin(answerid_min)]
    dupes = dupes[dupes[label_column].isin(answerid_min)]
    # Verify data integrity.
    assert questions[label_column].isin(dupes[label_column]).all()
    assert dupes[label_column].isin(questions[label_column]).all()
    # Report on the data.
    if show_output:
        print("Restrictions: min_text={}, min_dupes={}".format(min_text, min_dupes))
        print("Restricted text statistics:")
        print(
            pd.DataFrame(
                [
                    questions.Text.str.len().describe().rename("questions"),
                    dupes.Text.str.len().describe().rename("dupes"),
                ]
            )
        )
        print("\nRestricted duplication statistics:")
        print(
            pd.DataFrame(
                [dupes[label_column].value_counts().describe().rename("duplications")]
            )
        )
        print(
            "\nRestricted largest class: {:.2%}".format(
                dupes[label_column].value_counts().max() / dupes.shape[0]
            )
        )
    return dupes, label_column, questions


def save_data(
    balanced_pairs_test,
    balanced_pairs_train,
    dupes_test,
    dupes_test_path,
    outputs_path,
    questions,
    questions_path,
    show_output,
):
    """

    :param balanced_pairs_test:
    :param balanced_pairs_train:
    :param dupes_test:
    :param dupes_test_path:
    :param outputs_path:
    :param questions:
    :param questions_path:
    :param show_output:
    """
    os.makedirs(outputs_path, exist_ok=True)

    # Save the data.
    balanced_pairs_train_path = os.path.join(outputs_path, "balanced_pairs_train.tsv")
    print(f"Writing {balanced_pairs_train.shape[0]:,} to {balanced_pairs_train_path}")
    balanced_pairs_train.to_csv(
        balanced_pairs_train_path, sep="\t", header=True, index=False
    )
    balanced_pairs_test_path = os.path.join(outputs_path, "balanced_pairs_test.tsv")
    print(f"Writing {balanced_pairs_test.shape[0]:,} to {balanced_pairs_test_path}")
    balanced_pairs_test.to_csv(
        balanced_pairs_test_path, sep="\t", header=True, index=False
    )
    # Save original questions to be used for scoring later.
    if show_output:
        print("Writing {:,} to {}".format(questions.shape[0], questions_path))
    questions.to_csv(questions_path, sep="\t", header=True, index=False)
    # Save the test duplicate questions to be used with the scoring function.
    if show_output:
        print("Writing {:,} to {}".format(dupes_test.shape[0], dupes_test_path))
    dupes_test.to_csv(dupes_test_path, sep="\t", header=True, index=False)


def verify_data_integrity(answers, dupes, questions):
    """

    :param answers:
    :param dupes:
    :param questions:
    """
    # Verify data integrity.
    assert questions.AnswerId.isin(answers.index).all()
    assert answers.index.isin(questions.AnswerId).all()
    assert questions.AnswerId.isin(dupes.AnswerId).all()
    assert dupes.AnswerId.isin(questions.AnswerId).all()


def download_datasets(show_output=True):
    """

    :param show_output:
    :return:
    """
    # The output files path
    # URLs to original questions, duplicate questions, and answers.
    data_url = "https://bostondata.blob.core.windows.net/stackoverflow/{}"
    questions_url = data_url.format("orig-q.tsv.gz")
    dupes_url = data_url.format("dup-q.tsv.gz")
    answers_url = data_url.format("ans.tsv.gz")
    # Load datasets.
    questions = read_csv_gz(
        questions_url, names=("Id", "AnswerId", "Text0", "CreationDate")
    )
    dupes = read_csv_gz(dupes_url, names=("Id", "AnswerId", "Text0", "CreationDate"))
    answers = read_csv_gz(answers_url, names=("Id", "Text0"))

    if show_output:
        print(questions.iloc[0, 1])
        print(dupes[dupes.AnswerId == questions.iloc[0, 0]])
        print(answers.at[questions.iloc[0, 0], "Text0"])

    return answers, dupes, questions
