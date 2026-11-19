"""Word recognizer module (ASL Citizen, MediaPipe Holistic landmarks).

Importable API:

    from src.predictor import WordPredictor

Self-contained: it does not import anything from ../asl_landmark_app.
"""
__all__ = ["config", "vocab", "landmarks", "features", "extract", "dataset",
           "model", "train", "phrases", "predictor", "live"]
