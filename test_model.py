
from model import train_model
import numpy as np

def test_model_accuracy():
    model, accuracy = train_model()
    assert accuracy >= 0.80

def test_model_prediction():
    model, accuracy = train_model()

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    assert prediction[0] in [0, 1, 2]

def test_model_exists():
    model, accuracy = train_model()
    assert model is not None
