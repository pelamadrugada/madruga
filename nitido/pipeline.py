"""Orquestracao: junta rosto, deblur, realce e ampliacao.

A ideia central e simples e vale ser dita em voz alta:

* o **rosto** so e redimensionado — nenhuma deconvolucao, nenhum contraste
  local, nenhuma nitidez artificial toca nele;
* **todo o resto** passa por remocao de motion blur, contraste local e
  mascara de nitidez antes de subir para 5x;
* a emenda entre os dois e uma mascara suave, entao nao aparece costura.

O trabalho pesado (a imagem ja em 5x) e feito em faixas horizontais com
sobreposicao, para que o pico de memoria nao seja o da imagem inteira em
``float32``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import deblur as deblur_mod
from . import enhance as enh
from .faces import FaceBox, FaceDetector, face_mask

FACE_MODES = ("preserve", "gentle", "enhance", "off")

# Amostras que cada interpolador enxerga de cada lado do pixel de destino.
# Define o quanto as faixas precisam se sobrepor para nao deixar emenda.
INTERP_SUPPORT = {"lanczos": 4, "cubic": 2, "linear": 1, "nearest": 1}


@dataclass
class EnhanceConfig:
    """Todos os botoes do pipeline, com valores padrao ja utilizaveis."""

    # Ampliacao
    scale: float = 5.0
    interpolation: str = "lanczos"

    # Remocao de motion blur
    deblur_method: str = "wiener"  # wiener | rl | none
    deblur_strength: float = 0.85
    deblur_noise: float = 0.012
    deblur_iterations: int = 15
    deblur_min_confidence: float = 8.0
    deblur_max_length: int = 40

    # Realce do que nao e rosto
    denoise: float = 0.0
    clahe_clip: float = 1.6
    clahe_grid: int = 8
    sharpen_sigma: float = 1.6  # em pixels da SAIDA
    sharpen_amount: float = 0.7
    sharpen_threshold: float = 0.012

    # Rosto
    face_mode: str = "preserve"
    face_expand: float = 0.30
    face_feather: float = 0.30
    face_model: Optional[str] = None
    face_min_size_ratio: float = 0.02
    face_allow_download: bool = True

    # Execucao
    band_height: int = 256  # linhas da imagem ORIGINAL por faixa
    max_output_pixels: int = 250_000_000

    # Video
    codec: str = "mp4v"
    face_interval: int = 1
    blur_interval: int = 1

    def validate(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale precisa ser maior que zero")
        if self.face_mode not in FACE_MODES:
            raise ValueError(f"face-mode precisa ser um de {FACE_MODES}")
        if self.deblur_method not in ("wiener", "rl", "none"):
            raise ValueError("deblur precisa ser wiener, rl ou none")
        if self.interpolation not in enh.INTERPOLATIONS:
            raise ValueError(f"interpolacao precisa ser uma de {tuple(enh.INTERPOLATIONS)}")
        if self.band_height < 16:
            raise ValueError("band-height precisa ser >= 16")


@dataclass
class Report:
    """O que aconteceu com um quadro/imagem."""

    input_size: tuple = (0, 0)
    output_size: tuple = (0, 0)
    faces: int = 0
    blur_length: float = 0.0
    blur_angle: float = 0.0
    blur_confidence: float = 0.0
    deblur_applied: bool = False
    seconds: float = 0.0


def build_detector(cfg: EnhanceConfig) -> Optional[FaceDetector]:
    """Cria o detector, ou ``None`` quando o rosto nao precisa ser tratado."""
    if cfg.face_mode == "off":
        return None
    return FaceDetector(
        model=cfg.face_model,
        min_size_ratio=cfg.face_min_size_ratio,
        allow_download=cfg.face_allow_download,
    )


def enhance_image(
    bgr: np.ndarray,
    cfg: Optional[EnhanceConfig] = None,
    detector: Optional[FaceDetector] = None,
    boxes: Optional[Sequence[FaceBox]] = None,
    psf: Optional[np.ndarray] = None,
    blur_estimate: Optional[tuple] = None,
) -> tuple:
    """Processa uma imagem BGR uint8 e devolve ``(saida_uint8, Report)``.

    ``boxes``, ``psf`` e ``blur_estimate`` permitem reaproveitar deteccao e
    nucleo de borrao entre quadros de video — inclusive para que o relatorio
    do quadro continue mostrando o comprimento e o angulo corretos.
    """
    cfg = cfg or EnhanceConfig()
    cfg.validate()
    started = time.time()

    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    out_h = max(1, int(round(h * cfg.scale)))
    out_w = max(1, int(round(w * cfg.scale)))
    if out_h * out_w > cfg.max_output_pixels:
        raise MemoryError(
            f"saida teria {out_h * out_w / 1e6:.0f} MP, acima do limite de "
            f"{cfg.max_output_pixels / 1e6:.0f} MP. Reduza --scale ou aumente --max-pixels."
        )

    img = enh.to_float(bgr)

    # 1. Onde estao os rostos -----------------------------------------------
    if cfg.face_mode == "off":
        found: List[FaceBox] = []
    elif boxes is not None:
        found = list(boxes)
    elif detector is not None:
        found = detector.detect(bgr)
    else:
        found = []

    if cfg.face_mode in ("preserve", "gentle") and found:
        mask = face_mask((h, w), found, cfg.face_expand, cfg.face_feather)
    else:
        # Em "enhance"/"off" o rosto entra no mesmo tratamento do resto.
        mask = np.zeros((h, w), np.float32)
    allow = 1.0 - mask  # quanto de realce cada pixel pode receber

    # 2. Motion blur ---------------------------------------------------------
    deblurred, estimate = deblur_mod.deblur(
        img,
        method=cfg.deblur_method,
        strength=cfg.deblur_strength,
        noise=cfg.deblur_noise,
        iterations=cfg.deblur_iterations,
        min_confidence=cfg.deblur_min_confidence,
        max_length=cfg.deblur_max_length,
        psf=psf,
        estimate=blur_estimate,
    )

    # 3. O rosto vem de onde? ------------------------------------------------
    if cfg.face_mode == "preserve":
        face_source = img  # pixels originais, intocados
    else:
        face_source = deblurred  # "gentle": so o deblur alcanca o rosto

    # 4. Realce em resolucao nativa (barato) e composicao --------------------
    background = enh.bilateral_denoise(deblurred, cfg.denoise, weight=allow)
    background = enh.local_contrast(background, cfg.clahe_clip, cfg.clahe_grid, weight=allow)

    if found and cfg.face_mode in ("preserve", "gentle"):
        m3 = mask[:, :, None]
        native = face_source * m3 + background * (1.0 - m3)
    else:
        native = background

    # 5. Ampliacao + nitidez, em faixas --------------------------------------
    out = _upscale_and_sharpen(native, allow, cfg, out_h, out_w)

    report = Report(
        input_size=(w, h),
        output_size=(out_w, out_h),
        faces=len(found),
        blur_length=estimate.length,
        blur_angle=estimate.angle,
        blur_confidence=estimate.confidence,
        deblur_applied=estimate.used,
        seconds=time.time() - started,
    )
    return out, report


def _upscale_and_sharpen(
    native: np.ndarray,
    allow: np.ndarray,
    cfg: EnhanceConfig,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Amplia e aplica unsharp em faixas horizontais com sobreposicao.

    Os dois operadores envolvidos sao locais, entao basta que a sobreposicao
    cubra o alcance de ambos para a faixa sair identica ao que sairia da
    imagem inteira: ~3 sigma da Gaussiana (em pixels da saida) mais o suporte
    do filtro de interpolacao (4 amostras de cada lado, no Lanczos4).
    """
    h, w = native.shape[:2]
    out = np.empty((out_h, out_w, 3), np.uint8)

    gauss_reach = (3.0 * max(cfg.sharpen_sigma, 0.0) + 2.0) / max(cfg.scale, 1e-6)
    overlap = int(np.ceil(gauss_reach)) + INTERP_SUPPORT.get(cfg.interpolation, 4) + 2
    band = max(16, int(cfg.band_height))

    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        ya = max(0, y0 - overlap)
        yb = min(h, y1 + overlap)

        top_out = int(round(ya * cfg.scale))
        bottom_out = int(round(yb * cfg.scale))
        band_h = max(1, bottom_out - top_out)

        chunk = cv2.resize(
            native[ya:yb],
            (out_w, band_h),
            interpolation=enh.INTERPOLATIONS[cfg.interpolation],
        )
        weight = cv2.resize(allow[ya:yb], (out_w, band_h), interpolation=cv2.INTER_LINEAR)
        chunk = enh.unsharp(
            chunk,
            cfg.sharpen_sigma,
            cfg.sharpen_amount,
            cfg.sharpen_threshold,
            weight=weight,
        )

        dst0 = int(round(y0 * cfg.scale))
        dst1 = int(round(y1 * cfg.scale))
        out[dst0:dst1] = enh.to_uint8(chunk[dst0 - top_out : dst1 - top_out])

    return out


# --------------------------------------------------------------------------
# Arquivos
# --------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _write_params(path: str, quality: int) -> List[int]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if ext == ".webp":
        return [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    if ext == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    return []


def process_image_file(
    src: str,
    dst: str,
    cfg: Optional[EnhanceConfig] = None,
    quality: int = 95,
    detector: Optional[FaceDetector] = None,
) -> Report:
    """Processa um arquivo de imagem e grava o resultado."""
    cfg = cfg or EnhanceConfig()
    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"nao consegui ler a imagem: {src}")

    detector = detector if detector is not None else build_detector(cfg)
    out, report = enhance_image(image, cfg, detector=detector)

    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    if not cv2.imwrite(dst, out, _write_params(dst, quality)):
        raise IOError(f"nao consegui gravar a imagem: {dst}")
    return report


def process_video_file(
    src: str,
    dst: str,
    cfg: Optional[EnhanceConfig] = None,
    progress: Optional[Callable[[int, int, Report], None]] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """Processa um video quadro a quadro.

    Atencao: o OpenCV nao copia a trilha de audio. Se o video tem som,
    remultiplexe depois (por exemplo com ``ffmpeg -i saida.mp4 -i
    entrada.mp4 -c copy -map 0:v -map 1:a final.mp4``).
    """
    cfg = cfg or EnhanceConfig()
    cfg.validate()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"nao consegui abrir o video: {src}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        out_w = max(1, int(round(in_w * cfg.scale)))
        out_h = max(1, int(round(in_h * cfg.scale)))

        parent = os.path.dirname(os.path.abspath(dst))
        os.makedirs(parent, exist_ok=True)
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*cfg.codec), fps, (out_w, out_h))
        if not writer.isOpened():
            raise IOError(f"nao consegui abrir o gravador de video ({cfg.codec}) para {dst}")

        detector = build_detector(cfg)
        boxes: Optional[List[FaceBox]] = None
        psf: Optional[np.ndarray] = None
        estimate: Optional[tuple] = None
        index = 0
        deblurred_frames = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames and index >= max_frames:
                    break

                if detector is not None and (index % max(1, cfg.face_interval) == 0 or boxes is None):
                    boxes = detector.detect(frame)

                reestimar = index % max(1, cfg.blur_interval) == 0
                out, report = enhance_image(
                    frame,
                    cfg,
                    detector=None,
                    boxes=boxes,
                    psf=None if reestimar else psf,
                    blur_estimate=None if reestimar else estimate,
                )
                if reestimar:
                    # Quando o quadro nao tem borrao, zera: reaproveitar uma PSF
                    # antiga aplicaria deconvolucao onde a estimativa disse que nao.
                    psf = (
                        deblur_mod.motion_psf(report.blur_length, report.blur_angle)
                        if report.deblur_applied
                        else None
                    )
                    estimate = (
                        (report.blur_length, report.blur_angle, report.blur_confidence)
                        if report.deblur_applied
                        else None
                    )
                deblurred_frames += int(report.deblur_applied)

                writer.write(out)
                if progress is not None:
                    progress(index, total, report)
                index += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return {
        "frames": index,
        "fps": fps,
        "input_size": (in_w, in_h),
        "output_size": (out_w, out_h),
        "deblurred_frames": deblurred_frames,
        "has_audio": False,
    }
