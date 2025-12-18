from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, model):
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )
    return pipeline