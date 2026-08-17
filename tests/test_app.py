import numpy as np

from app.predict import predict_signal_array
from app.preprocess import FEATURE_COLUMNS, extract_features


def _make_example_signal() -> np.ndarray:
    t = np.linspace(0, 1, 512)
    signal = 128.0 + 15.0 * np.sin(2 * np.pi * 4 * t) + 5.0 * np.sin(2 * np.pi * 20 * t)
    return signal.astype(float)


def test_extract_features_returns_expected_shape():
    signal = _make_example_signal()
    features = extract_features(signal)

    assert set(features.keys()) == set(FEATURE_COLUMNS)
    for value in features.values():
        assert np.isfinite(value)


def test_prediction_returns_valid_class():
    signal = _make_example_signal()
    result = predict_signal_array(signal)

    assert result["prediction"] in (0, 1)
    assert result["label"] in {"Parasite", "Treffer"}
    assert 0.0 <= result["confidence"] <= 1.0
