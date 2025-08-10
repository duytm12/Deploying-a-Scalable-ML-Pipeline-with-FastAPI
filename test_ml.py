import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test that the train_model function returns a trained RandomForestClassifier
    """
    # Create sample data
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)

    # Train the model
    model = train_model(X_train, y_train)

    # Check that the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    # Check that the model is fitted
    assert hasattr(model, 'classes_')


# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    Test that the inference function returns predictions of the correct shape
    """
    # Create a simple model and data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.random.rand(50, 3)
    y_train = np.random.randint(0, 2, 50)
    model.fit(X_train, y_train)

    # Test data
    X_test = np.random.rand(10, 3)

    # Get predictions
    preds = inference(model, X_test)

    # Check that predictions have the correct shape
    assert preds.shape == (10,)
    # Check that predictions are binary (0 or 1)
    assert all(pred in [0, 1] for pred in preds)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that the compute_model_metrics function returns expected metric values
    """
    # Create sample data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check that metrics are floats between 0 and 1
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
