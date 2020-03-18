"""
ai-utilities - machine_learning/duplicate_model.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pandas as pd
import joblib


class DuplicateModel:
    """ Create Copy of Model """

    questions_cols = ["Id", "AnswerId", "Text"]
    dup_col = "Text_x"
    id_col = "Id_y"
    answer_id_col = "AnswerId_y"
    orig_col = "Text_y"
    feature_cols = [dup_col, orig_col]
    probabilities_col = "probabilities"

    def __init__(self, model_path, questions_path):
        self.model_path = model_path
        self.questions_path = questions_path
        self.model = joblib.load(model_path)
        self.questions = pd.read_csv(questions_path, sep="\t", encoding="latin1")
        self.questions = self.questions[self.questions_cols]
        self.questions.columns = [self.id_col, self.answer_id_col, self.orig_col]

    def score(self, text):
        """
        Score Text Input

        :param text: Text Input
        :return: Input with Scores
        """
        # Create a scoring dataframe.
        test = self.questions.copy()
        test[self.dup_col] = text
        test_x = test[self.feature_cols]

        # Score the text.
        test[self.probabilities_col] = self.model.predict_proba(test_x)[:, 1]

        # Order the data by descending probability.
        test.sort_values(by=self.probabilities_col, ascending=False, inplace=True)

        # Extract the original question ids, answer ids, and probabilities.
        scores = test[[self.id_col, self.answer_id_col, self.probabilities_col]]
        pairs = [x[1:] for x in scores.itertuples()]

        # Return the result.
        return pairs
