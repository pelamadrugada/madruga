"""Realce de detalhe e redimensionamento.

Todas as funcoes recebem e devolvem BGR ``float32`` em ``[0, 1]`` e aceitam
um mapa ``weight`` opcional (mesma altura/largura, ``[0, 1]``) que diz
*quanto* de cada efeito aplicar em cada pixel. E por ai que o rosto fica de
fora: o pipeline passa ``weight = 1 - mascara_do_rosto``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

INTERPOLATIONS = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _blend(base: np.ndarray, processed: np.ndarray, weight: Optional[np.ndarray]) -> np.ndarray:
    if weight is None:
        return processed
    w = weight if weight.ndim == 3 else weight[:, :, None]
    return base * (1.0 - w) + processed * w


def upscale(img: np.ndarray, scale: float, interpolation: str = "lanczos") -> np.ndarray:
    """Amplia por ``scale`` mantendo dtype e numero de canais."""
    if scale == 1.0:
        return img
    interp = INTERPOLATIONS.get(interpolation)
    if interp is None:
        raise ValueError(f"interpolacao desconhecida: {interpolation!r}")
    h, w = img.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    # INTER_AREA e o certo para reduzir; Lanczos/cubic so servem para ampliar.
    if scale < 1.0:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def local_contrast(
    bgr: np.ndarray,
    clip: float = 1.6,
    grid: int = 8,
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Contraste local (CLAHE) na luminancia — puxa textura sem estourar cor."""
    if clip <= 0:
        return bgr
    ycrcb = cv2.cvtColor(np.clip(bgr, 0.0, 1.0), cv2.COLOR_BGR2YCrCb)
    luma = ycrcb[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    equalized = clahe.apply((luma * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    ycrcb[:, :, 0] = equalized
    processed = np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), 0.0, 1.0)
    return np.clip(_blend(bgr, processed, weight), 0.0, 1.0)


def unsharp(
    bgr: np.ndarray,
    sigma: float = 1.4,
    amount: float = 0.7,
    threshold: float = 0.012,
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mascara de nitidez com limiar.

    O limiar evita transformar ruido de sensor em granulado: so diferencas
    acima de ``threshold`` (em unidades de ``[0, 1]``) sao amplificadas.
    """
    if amount <= 0 or sigma <= 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    detail = bgr - blurred
    if threshold > 0:
        keep = (np.abs(detail) >= threshold).astype(np.float32)
        detail = detail * keep
    processed = np.clip(bgr + float(amount) * detail, 0.0, 1.0)
    return np.clip(_blend(bgr, processed, weight), 0.0, 1.0)


def bilateral_denoise(
    bgr: np.ndarray,
    strength: float = 0.0,
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Suaviza ruido preservando bordas, antes de qualquer realce."""
    if strength <= 0:
        return bgr
    sigma_color = float(np.clip(strength, 0.0, 1.0)) * 0.15
    processed = cv2.bilateralFilter(bgr, d=5, sigmaColor=sigma_color, sigmaSpace=5)
    return np.clip(_blend(bgr, processed, weight), 0.0, 1.0)


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 BGR em ``[0, 1]``."""
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float32 BGR em ``[0, 1]`` -> uint8 BGR."""
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
