from sklearn.pipeline import Pipeline


def get_pipeline(transformer, model):
    steps = [('transformer', transformer), ('model', model)]
    pipeline = Pipeline(steps=steps)

    return pipeline
