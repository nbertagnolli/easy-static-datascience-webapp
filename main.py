from typing import Any, Dict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, r2_score
from functools import partial
from io import StringIO


def main(data: str, response_column: str) -> Dict[str, Any]:
    df = pd.read_table(StringIO(data), sep=",")
    column_names = list(df.columns)
    print(df.head())

    columns_to_train = list(set(column_names).difference(set([response_column])))

    # Auto Determine Categorical, or continuous
    column_transformations = []
    for column in columns_to_train:
        if df.dtypes[column] == "float64":
            column_transformations.append(
                (f"scaled_{column}", StandardScaler(), [column])
            )
        else:
            column_transformations.append(
                (f"one_hot_{column}", OneHotEncoder(), [column])
            )

    ct = ColumnTransformer(column_transformations)

    # Auto determine regression or classification
    is_regression = df.dtypes[response_column] == "float64"
    model = RandomForestRegressor() if is_regression else RandomForestClassifier()

    model_pipeline = Pipeline([("data_transforms", ct), ("model", model)])

    # Train the model
    model_pipeline.fit(df[columns_to_train], df[response_column])

    # Make predictions on the training data
    preds = model_pipeline.predict(df[columns_to_train])

    # Get a classification report
    score_fn = (
        r2_score if is_regression else partial(classification_report, output_dict=True)
    )
    score = score_fn(df[response_column], preds)

    # Remove accuracy if they exist because they make the plots look bad.
    score.pop("accuracy", None)

    # Put into the format for plotly
    class_names = list(score.keys())
    print(class_names)
    score_names = list(score[class_names[0]].keys())
    results = []
    for score_name in score_names:
        if score_name != "support":
            results.append(
                {
                    "x": class_names,
                    "y": [score[label][score_name] for label in class_names],
                    "name": score_name,
                    "type": "bar",
                }
            )

    return {"scores": results}
