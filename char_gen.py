"""
Utilidades para modelado de lenguaje a nivel de carácter con LSTM:
- Construcción de vocabulario y datos (secuencias)
- One-hot encoding
- Creación y entrenamiento del modelo LSTM
- Muestreo con temperatura y generación de texto
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


__all__ = [
    "set_seeds",
    "build_char_vocab",
    "make_char_sequences",
    "one_hot_encode",
    "build_lstm_char_model",
    "sample_with_temperature",
    "generate_text",
]


def set_seeds(seed: int = 42) -> None:
    """
    Fija semillas para reproducibilidad en NumPy y TensorFlow.

    Args:
        seed: Entero para inicializar RNGs.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_char_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Crea vocabulario de caracteres y diccionarios índice<->carácter.

    Args:
        text: Corpus en texto plano.

    Returns:
        chars: Lista ordenada de caracteres únicos.
        stoi: Dict char->índice.
        itos: Dict índice->char.
    """
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def make_char_sequences(text: str, seq_len: int, stoi: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte el corpus en pares (secuencia, siguiente_caracter) a nivel de carácter.

    Args:
        text: Corpus en texto plano.
        seq_len: Longitud fija de cada secuencia.
        stoi: Diccionario char->índice.

    Returns:
        X_idx: np.ndarray int32 de shape (n_samples, seq_len) con índices de caracteres.
        y_idx: np.ndarray int32 de shape (n_samples,) con el índice del siguiente carácter.
    """
    X_idx, y_idx = [], []
    for i in range(len(text) - seq_len):
        seq = text[i : i + seq_len]
        nxt = text[i + seq_len]
        X_idx.append([stoi[c] for c in seq])
        y_idx.append(stoi[nxt])
    return np.array(X_idx, dtype="int32"), np.array(y_idx, dtype="int32")


def one_hot_encode(X_idx: np.ndarray, y_idx: np.ndarray, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica one-hot a X e y (a nivel de carácter).

    Args:
        X_idx: Índices de entrada (n_samples, seq_len).
        y_idx: Índices objetivo (n_samples,).
        vocab_size: Tamaño del vocabulario de caracteres.

    Returns:
        X_oh: One-hot de entradas (n_samples, seq_len, vocab_size), float32.
        y_oh: One-hot de objetivos (n_samples, vocab_size), float32.
    """
    X_oh = to_categorical(X_idx, num_classes=vocab_size).astype("float32")
    y_oh = to_categorical(y_idx, num_classes=vocab_size).astype("float32")
    return X_oh, y_oh


def build_lstm_char_model(
    seq_len: int,
    vocab_size: int,
    units: int = 128,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Construye un modelo LSTM para modelado de lenguaje a nivel carácter (one-hot).

    Arquitectura:
        LSTM(units) -> Dense(vocab_size, softmax)

    Args:
        seq_len: Longitud de las secuencias de entrada.
        vocab_size: Tamaño del vocabulario (número de caracteres).
        units: Unidades LSTM.
        learning_rate: Tasa de aprendizaje para Adam.

    Returns:
        Modelo Keras compilado con pérdida 'categorical_crossentropy'.
    """
    model = Sequential(
        [
            LSTM(units, input_shape=(seq_len, vocab_size)),
            Dense(vocab_size, activation="softmax"),
        ],
        name="char_lstm_language_model",
    )
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy")
    return model


def sample_with_temperature(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Selecciona un índice de la distribución 'probs' aplicando temperatura.

    Args:
        probs: Probabilidades (vector 1D que suma 1).
        temperature: Control de entropía. <1 hace la dist. más “conservadora”; >1 más “creativa”.

    Returns:
        Índice entero muestreado.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(np.maximum(probs, 1e-9)) / temperature
    logits -= np.max(logits)  # estabilidad numérica
    p = np.exp(logits)
    p /= np.sum(p)
    return int(np.random.choice(len(p), p=p))


def generate_text(
    model: Model,
    seed: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    seq_len: int,
    vocab_size: int,
    length: int = 300,
    temperature: float = 0.8,
    lowercase: bool = True,
) -> str:
    """
    Genera texto carácter a carácter a partir de una semilla.

    Args:
        model: Modelo LSTM entrenado.
        seed: Texto semilla. Se recorta/ajusta a seq_len.
        stoi: Diccionario char->índice.
        itos: Diccionario índice->char.
        seq_len: Longitud de ventana de contexto.
        vocab_size: Tamaño del vocabulario.
        length: Número de caracteres a generar.
        temperature: Temperatura de muestreo.
        lowercase: Si True, convierte la semilla a minúsculas.

    Returns:
        Texto generado (seed + caracteres generados).
    """
    if lowercase:
        seed = seed.lower()
    seed = seed[:seq_len].ljust(seq_len)
    generated = seed

    for _ in range(length):
        x_idx = np.array([[stoi.get(c, 0) for c in generated[-seq_len:]]], dtype="int32")
        x_oh = to_categorical(x_idx, num_classes=vocab_size).astype("float32")
        preds = model.predict(x_oh, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        generated += itos[next_idx]
    return generated
