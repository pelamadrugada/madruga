"""Deteccao de rostos e construcao da mascara de protecao.

A mascara e um mapa suave em ``[0, 1]`` onde ``1`` significa "isto e rosto,
nao mexa" e ``0`` significa "pode processar a vontade". As bordas sao
suavizadas com um desfoque proporcional ao tamanho do rosto para que a
composicao final nao deixe uma emenda visivel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .models import ModelUnavailable, resolve_model

# Lado maximo usado para *detectar*. Imagens maiores sao reduzidas so para a
# deteccao e as caixas voltam para a escala original. Detectar em 40 MP e
# desperdicio: rostos continuam sendo rostos a 1024 px.
DETECTION_MAX_SIDE = 1024


@dataclass(frozen=True)
class FaceBox:
    """Caixa de um rosto em pixels da imagem original."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    def scaled(self, factor: float) -> "FaceBox":
        return FaceBox(
            int(round(self.x * factor)),
            int(round(self.y * factor)),
            int(round(self.w * factor)),
            int(round(self.h * factor)),
            self.score,
        )


class FaceDetector:
    """Detector de rostos, com YuNet na frente e Haar como plano B.

    * **YuNet** (``face_detection_yunet_2023mar.onnx``): padrao. Pega rostos
      pequenos e de perfil. O arquivo e resolvido por :mod:`nitido.models`
      (flag, variavel de ambiente, cache ou download unico).
    * **Haar cascade**: usado apenas em instalacoes com OpenCV 4.x, que ainda
      empacotam os XMLs. O OpenCV 5 removeu tanto os arquivos quanto a classe
      ``CascadeClassifier``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        score_threshold: float = 0.6,
        min_size_ratio: float = 0.02,
        allow_download: bool = True,
    ) -> None:
        self.score_threshold = score_threshold
        self.min_size_ratio = min_size_ratio
        self._yunet = None
        self._cascades: List["cv2.CascadeClassifier"] = []

        model_path = resolve_model(model, allow_download=allow_download)
        if model_path:
            self._yunet = cv2.FaceDetectorYN.create(
                model_path, "", (320, 320), score_threshold
            )
            self.backend = "yunet"
            self.model_path = model_path
            return

        self._load_cascades()
        if not self._cascades:
            raise ModelUnavailable(
                "sem detector de rostos disponivel: o OpenCV instalado nao traz "
                "classificadores Haar e o modelo YuNet nao foi encontrado. "
                "Rode com --face-model CAMINHO/face_detection_yunet_2023mar.onnx, "
                "ou com --face-mode off para processar sem proteger o rosto."
            )
        self.backend = "haar"
        self.model_path = None

    def _load_cascades(self) -> None:
        if not hasattr(cv2, "CascadeClassifier"):
            return  # OpenCV 5.x removeu a API de cascatas
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if not data_dir or not os.path.isdir(data_dir):
            return
        for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                continue
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self._cascades.append(cascade)

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        """Devolve os rostos encontrados em ``bgr`` (uint8, BGR)."""
        h, w = bgr.shape[:2]
        shrink = min(1.0, DETECTION_MAX_SIDE / float(max(h, w)))
        if shrink < 1.0:
            small = cv2.resize(bgr, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        boxes = self._detect_yunet(small) if self._yunet is not None else self._detect_haar(small)

        back = 1.0 / shrink if shrink > 0 else 1.0
        min_side = self.min_size_ratio * max(h, w)
        out: List[FaceBox] = []
        for box in boxes:
            full = box.scaled(back) if shrink < 1.0 else box
            if max(full.w, full.h) < min_side:
                continue
            out.append(_clamp_box(full, w, h))
        return _merge_overlapping(out)

    # -- backends ---------------------------------------------------------

    def _detect_yunet(self, bgr: np.ndarray) -> List[FaceBox]:
        h, w = bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(bgr)
        if faces is None:
            return []
        return [
            FaceBox(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1]))
            for f in faces
            if float(f[-1]) >= self.score_threshold
        ]

    def _detect_haar(self, bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        found: List[FaceBox] = []
        for cascade in self._cascades:
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            found.extend(FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in rects)
        # O cascade de perfil so pega um lado; espelhar cobre o outro.
        if len(self._cascades) > 1:
            flipped = cv2.flip(gray, 1)
            width = gray.shape[1]
            rects = self._cascades[-1].detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            found.extend(
                FaceBox(int(width - x - w), int(y), int(w), int(h)) for x, y, w, h in rects
            )
        return found


def _clamp_box(box: FaceBox, w: int, h: int) -> FaceBox:
    x = max(0, min(box.x, w - 1))
    y = max(0, min(box.y, h - 1))
    return FaceBox(x, y, max(1, min(box.w, w - x)), max(1, min(box.h, h - y)), box.score)


def _merge_overlapping(boxes: Sequence[FaceBox], iou_threshold: float = 0.35) -> List[FaceBox]:
    """Funde caixas muito sobrepostas (frontal + perfil pegam o mesmo rosto)."""
    kept: List[FaceBox] = []
    for box in sorted(boxes, key=lambda b: b.w * b.h, reverse=True):
        if any(_iou(box, other) > iou_threshold for other in kept):
            continue
        kept.append(box)
    return kept


def _iou(a: FaceBox, b: FaceBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    return inter / float(a.w * a.h + b.w * b.h - inter)


def face_mask(
    shape: Sequence[int],
    boxes: Sequence[FaceBox],
    expand: float = 0.30,
    feather: float = 0.30,
) -> np.ndarray:
    """Monta a mascara suave de protecao dos rostos.

    Parameters
    ----------
    shape:
        ``(altura, largura)`` da imagem.
    boxes:
        Rostos detectados.
    expand:
        Quanto crescer a elipse alem da caixa detectada, em fracao do lado.
        Cobre cabelo, queixo e orelhas, que o detector costuma cortar.
    feather:
        Largura da transicao, em fracao do tamanho do rosto.

    Returns
    -------
    np.ndarray
        Mapa ``float32`` em ``[0, 1]`` com a altura/largura pedidas.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.float32)
    if not boxes:
        return mask

    for box in boxes:
        cx = box.x + box.w / 2.0
        # O centro sobe um pouco: a testa/cabelo pesa mais que o pescoco.
        cy = box.y + box.h * 0.46
        ax = box.w * 0.5 * (1.0 + expand)
        ay = box.h * 0.5 * (1.0 + expand + 0.18)
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(ax))), max(1, int(round(ay)))),
            0,
            0,
            360,
            1.0,
            thickness=-1,
        )

    mean_side = float(np.mean([max(b.w, b.h) for b in boxes]))
    sigma = max(1.0, mean_side * feather * 0.5)
    ksize = int(sigma * 4) | 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
