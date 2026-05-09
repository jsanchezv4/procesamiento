"""
Módulo utilitario para construir y entrenar modelos secuenciales (RNN, LSTM, GRU)
sobre datos sintéticos de ejemplo.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def set_seeds(seed: int = 42) -> None:
    """
    Fija semillas de aleatoriedad para Numpy y TensorFlow con fines de reproducibilidad.

    Args:
        seed: Entero para inicializar los generadores de números aleatorios.

    Returns:
        None
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_synthetic_regression_data(
    batch: int = 256,
    timesteps: int = 20,
    features: int = 8,
    y_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset sintético para regresión con entrada secuencial.

    Cada muestra tiene forma (timesteps, features). El objetivo y es una combinación
    lineal suave de algunos pasos y características, con ruido opcional.

    Args:
        batch: Número de secuencias a generar.
        timesteps: Longitud temporal de cada secuencia.
        features: Número de características por paso temporal.
        y_noise: Desviación estándar del ruido gaussiano agregado al objetivo.

    Returns:
        tuple (X, y)
            X: np.ndarray de shape (batch, timesteps, features), dtype float32.
            y: np.ndarray de shape (batch, 1), dtype float32.
    """
    X = np.random.randn(batch, timesteps, features).astype("float32")

    # Construimos un objetivo con dependencias temporales: promedio ponderado
    # de las últimas posiciones y algunas features, para que no sea completamente aleatorio.
    weights_t = np.linspace(0.2, 1.0, timesteps).reshape(1, timesteps, 1)  # más peso al final
    weights_f = np.linspace(1.0, 0.5, features).reshape(1, 1, features)
    y_raw = (X * weights_t * weights_f).sum(axis=(1, 2), keepdims=True).astype("float32")

    if y_noise > 0:
        y_raw += np.random.normal(0.0, y_noise, size=y_raw.shape).astype("float32")

    # Normalizamos para estabilizar la escala de pérdida
    y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)
    return X, y.astype("float32")

def train_test_split_seq(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """
    Divide (X, y) en conjuntos de entrenamiento y prueba conservando el orden (útil en series).
    Para un split aleatorio, baraja con una permutación antes de dividir.

    Args:
        X: Tensores de entrada con shape (n_samples, timesteps, features).
        y: Objetivos con shape (n_samples, 1).
        test_size: Proporción para el conjunto de prueba.
        seed: Semilla para reproducibilidad si se aplica barajado.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Opción A (ordenada): split por índice
    n = len(X)
    n_test = int(n * test_size)
    split = n - n_test
    return X[:split], X[split:], y[:split], y[split:]

def build_seq_model(
    kind: Literal["rnn", "lstm", "gru"],
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Construye un modelo Keras secuencial de tipo RNN simple, LSTM o GRU
    para tareas de regresión sobre secuencias.

    Arquitectura: [Recurrente(units)] -> Dropout(dropout) -> Dense(1)

    Args:
        kind: Tipo de celda recurrente a usar: "rnn", "lstm" o "gru".
        input_shape: Tupla (timesteps, features) de la entrada.
        units: Número de unidades ocultas de la capa recurrente.
        dropout: Proporción de neuronas a desactivar tras la capa recurrente.
        learning_rate: Tasa de aprendizaje del optimizador Adam.

    Returns:
        Modelo Keras compilado (pérdida MSE, optimizador Adam).
    """
    model = Sequential(name=f"{kind.upper()}_regressor")

    if kind == "rnn":
        model.add(SimpleRNN(units, input_shape=input_shape, activation="tanh"))
    elif kind == "lstm":
        model.add(LSTM(units, input_shape=input_shape))
    elif kind == "gru":
        model.add(GRU(units, input_shape=input_shape))
    else:
        raise ValueError('kind debe ser "rnn", "lstm" o "gru".')

    if dropout and dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def summarize_models(models: Dict[str, Model]) -> None:
    """
    Imprime el resumen de múltiples modelos Keras con un encabezado identificador.

    Args:
        models: Diccionario {nombre: modelo}.

    Returns:
        None
    """
    for name, m in models.items():
        print(f"\n{name} summary:")
        m.summary()


def fit_models(
    models: Dict[str, Model],
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 3,
    batch_size: int | None = None,
    verbose: int = 0,
) -> Dict[str, tf.keras.callbacks.History]:
    """
    Entrena varios modelos sobre los mismos datos y devuelve sus historiales.

    Args:
        models: Diccionario {nombre: modelo}.
        X: Tensores de entrada con shape (batch, timesteps, features).
        y: Objetivos con shape (batch, 1) u otra compatible.
        epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño de lote; si es None, Keras elige por defecto.
        verbose: Verbosidad de Keras (0, 1 o 2).

    Returns:
        Diccionario {nombre: History} con los historiales de entrenamiento.
    """
    histories: Dict[str, tf.keras.callbacks.History] = {}
    for name, model in models.items():
        histories[name] = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return histories

def predict_models(models: Dict[str, Model], X: np.ndarray, verbose: int = 0) -> Dict[str, np.ndarray]:
    """
    Realiza predicciones con múltiples modelos sobre el mismo conjunto X.

    Args:
        models: Diccionario {nombre: modelo}.
        X: Tensores de entrada con shape (batch, timesteps, features).
        verbose: Verbosidad de Keras al predecir.

    Returns:
        Diccionario {nombre: y_pred} con arrays de predicción.
    """
    preds: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        preds[name] = model.predict(X, verbose=verbose)
    return preds

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de regresión básicas: MSE, MAE y R^2.

    Args:
        y_true: Valores verdaderos, shape (n_samples, 1) o compatible.
        y_pred: Predicciones, shape (n_samples, 1) o compatible.

    Returns:
        Diccionario con 'MSE', 'MAE' y 'R2'.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # R^2 = 1 - SSE/SST
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    r2 = float(1.0 - sse / sst)
    return {"MSE": mse, "MAE": mae, "R2": r2}
