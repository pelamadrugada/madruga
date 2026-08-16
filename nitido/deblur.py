"""Estimativa e remocao de motion blur (borrao de movimento).

O borrao de movimento linear e uma convolucao da imagem nitida por um
"risco" — um segmento de reta com comprimento `L` e angulo `theta`. Aqui:

1. :func:`estimate_motion_psf` estima `L` e `theta` pelo cepstro da imagem;
2. :func:`motion_psf` reconstroi o nucleo (PSF);
3. :func:`wiener_deconvolve` / :func:`richardson_lucy` desfazem a convolucao.

Tudo trabalha em ``float32`` no intervalo ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class BlurEstimate:
    """Resultado da estimativa do borrao."""

    length: float
    angle: float
    confidence: float
    used: bool = False
    method: str = "none"
    psf: Optional[np.ndarray] = field(default=None, repr=False)


def motion_psf(length: float, angle_deg: float) -> np.ndarray:
    """Nucleo (PSF) de um borrao de movimento linear, normalizado.

    ``angle_deg`` segue a convencao matematica: 0 aponta para a direita,
    90 para cima.
    """
    steps = max(int(round(length)), 1)
    size = steps if steps % 2 == 1 else steps + 1
    kernel = np.zeros((size, size), np.float32)
    center = size // 2
    if steps == 1:
        kernel[center, center] = 1.0
        return kernel

    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    # Superamostra o segmento para que angulos diagonais nao fiquem falhados.
    for t in np.linspace(-length / 2.0, length / 2.0, steps * 8):
        x = int(round(center + dx * t))
        y = int(round(center + dy * t))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] += 1.0

    total = float(kernel.sum())
    if total <= 0:
        kernel[center, center] = 1.0
        return kernel
    return kernel / total


def estimate_motion_psf(
    gray: np.ndarray,
    min_length: int = 4,
    max_length: int = 40,
) -> Tuple[float, float, float]:
    """Estima comprimento, angulo e confianca do borrao pelo cepstro.

    A convolucao vira soma no dominio do log-espectro, e o log-espectro de
    um borrao linear e periodico: no cepstro isso aparece como um vale
    marcado a uma distancia igual ao comprimento do borrao, na direcao do
    movimento.

    Returns
    -------
    (length, angle_deg, confidence)
        ``confidence`` e quantos desvios-padrao o vale esta abaixo da media
        do anel de busca. Valores acima de ~3 indicam borrao direcional
        real; abaixo disso a imagem provavelmente so esta fora de foco.
    """
    h, w = gray.shape[:2]
    side = int(min(512, h, w))
    if side < 32:
        return 0.0, 0.0, 0.0
    max_length = int(min(max_length, side // 2 - 2))
    if max_length <= min_length:
        return 0.0, 0.0, 0.0

    y0 = (h - side) // 2
    x0 = (w - side) // 2
    patch = gray[y0 : y0 + side, x0 : x0 + side].astype(np.float32)
    patch = patch - float(patch.mean())
    window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)

    spectrum = np.fft.fft2(patch * window)
    log_mag = np.log1p(np.abs(spectrum))
    cepstrum = np.real(np.fft.ifft2(log_mag))
    cepstrum = np.fft.fftshift(cepstrum)

    center = side // 2
    yy, xx = np.mgrid[0:side, 0:side]
    radius = np.hypot(yy - center, xx - center)
    ring = (radius >= min_length) & (radius <= max_length)
    if not ring.any():
        return 0.0, 0.0, 0.0

    values = cepstrum[ring]
    mean = float(values.mean())
    std = float(values.std()) + 1e-8
    masked = np.where(ring, cepstrum, np.inf)
    idx = np.unravel_index(int(np.argmin(masked)), masked.shape)
    dy = float(idx[0] - center)
    dx = float(idx[1] - center)

    length = float(np.hypot(dy, dx))
    angle = float(np.degrees(np.arctan2(-dy, dx)) % 180.0)
    confidence = (mean - float(cepstrum[idx])) / std
    return length, angle, confidence


def _pad_for_psf(img: np.ndarray, psf: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Espelha as bordas para a FFT nao enrolar o borrao de um lado no outro."""
    py = psf.shape[0]
    px = psf.shape[1]
    padded = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REFLECT_101)
    return padded, py, px


def _psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    canvas[:kh, :kw] = psf
    canvas = np.roll(canvas, -(kh // 2), axis=0)
    canvas = np.roll(canvas, -(kw // 2), axis=1)
    return np.fft.rfft2(canvas)


def wiener_deconvolve(channel: np.ndarray, psf: np.ndarray, noise: float = 0.01) -> np.ndarray:
    """Deconvolucao de Wiener de um canal ``float32``.

    ``noise`` e a razao ruido/sinal: mais alto significa resultado mais
    conservador (menos anel, menos nitidez).
    """
    padded, py, px = _pad_for_psf(channel, psf)
    otf = _psf2otf(psf, padded.shape)
    spectrum = np.fft.rfft2(padded)
    filtered = np.conj(otf) / (np.abs(otf) ** 2 + max(noise, 1e-6)) * spectrum
    restored = np.fft.irfft2(filtered, s=padded.shape).astype(np.float32)
    return restored[py : py + channel.shape[0], px : px + channel.shape[1]]


def richardson_lucy(channel: np.ndarray, psf: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Deconvolucao de Richardson-Lucy: mais lenta, menos anel que Wiener."""
    padded, py, px = _pad_for_psf(channel, psf)
    flipped = psf[::-1, ::-1].copy()
    estimate = np.clip(padded, 1e-4, None)
    for _ in range(max(1, iterations)):
        blurred = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT_101)
        ratio = padded / np.maximum(blurred, 1e-4)
        estimate = estimate * cv2.filter2D(ratio, -1, flipped, borderType=cv2.BORDER_REFLECT_101)
        np.clip(estimate, 0.0, 4.0, out=estimate)
    return estimate[py : py + channel.shape[0], px : px + channel.shape[1]].astype(np.float32)


def deblur(
    bgr: np.ndarray,
    method: str = "wiener",
    strength: float = 1.0,
    noise: float = 0.012,
    iterations: int = 15,
    min_confidence: float = 8.0,
    max_length: int = 40,
    psf: Optional[np.ndarray] = None,
    estimate: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, BlurEstimate]:
    """Remove motion blur de uma imagem BGR ``float32`` em ``[0, 1]``.

    A deconvolucao roda so na luminancia (canal Y de YCrCb). Croma quase nao
    carrega detalhe e deconvoluir cor e receita de franja colorida.

    ``strength`` mistura o resultado com o original (``0`` = nao mexe,
    ``1`` = deconvolucao cheia). ``psf`` permite passar um nucleo pronto —
    util em video, para reaproveitar a estimativa entre quadros.
    """
    if method == "none" or strength <= 0.0:
        return bgr, BlurEstimate(0.0, 0.0, 0.0, used=False, method="none")

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    luma = ycrcb[:, :, 0]

    if psf is None:
        if estimate is None:
            estimate = estimate_motion_psf(luma, max_length=max_length)
        length, angle, confidence = estimate
        if confidence < min_confidence or length < 2.0:
            return bgr, BlurEstimate(length, angle, confidence, used=False, method=method)
        psf = motion_psf(length, angle)
    else:
        length, angle, confidence = estimate or (float(psf.shape[0]), 0.0, float("inf"))

    if method == "rl":
        restored = richardson_lucy(luma, psf, iterations)
    elif method == "wiener":
        restored = wiener_deconvolve(luma, psf, noise)
    else:
        raise ValueError(f"metodo de deblur desconhecido: {method!r}")

    blend = float(np.clip(strength, 0.0, 1.0))
    ycrcb[:, :, 0] = np.clip(luma * (1.0 - blend) + restored * blend, 0.0, 1.0)
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return (
        np.clip(out, 0.0, 1.0),
        BlurEstimate(length, angle, confidence, used=True, method=method, psf=psf),
    )
