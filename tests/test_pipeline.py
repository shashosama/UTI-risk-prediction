from src.preprocess import load_data

def test_data_loaded():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_train.shape[1] > 0
