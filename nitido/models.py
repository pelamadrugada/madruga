"""Localizacao (e download opcional) do modelo de deteccao de rostos.

O OpenCV 5 removeu os classificadores Haar que vinham empacotados, entao o
detector padrao passou a ser o **YuNet** — um ONNX de ~230 KB do repositorio
oficial ``opencv/opencv_zoo``. Ele nao vem com o ``pip install``, e resolvido
assim, na ordem:

1. caminho passado em ``--face-model``;
2. variavel de ambiente ``NITIDO_FACE_MODEL``;
3. arquivo ja baixado no cache (``~/.cache/nitido``, ou ``NITIDO_CACHE_DIR``);
4. download unico, se permitido.

Sem modelo nao ha deteccao — e sem deteccao nao ha como proteger o rosto.
Nesse caso o programa para e explica o que fazer, em vez de processar o rosto
sem querer.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from typing import Optional

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

ENV_MODEL = "NITIDO_FACE_MODEL"
ENV_CACHE = "NITIDO_CACHE_DIR"


class ModelUnavailable(RuntimeError):
    """Nao foi possivel obter o modelo de deteccao de rostos."""


def cache_dir() -> str:
    override = os.environ.get(ENV_CACHE)
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "nitido")


def cached_model_path() -> str:
    return os.path.join(cache_dir(), MODEL_FILENAME)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(destination: Optional[str] = None, timeout: float = 60.0) -> str:
    """Baixa o YuNet para o cache e confere o hash. Devolve o caminho."""
    destination = destination or cached_model_path()
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(destination)))
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response:
            with open(tmp_path, "wb") as handle:
                while True:
                    block = response.read(1 << 16)
                    if not block:
                        break
                    handle.write(block)

        got = sha256(tmp_path)
        if got != MODEL_SHA256:
            raise ModelUnavailable(
                f"o modelo baixado nao confere (sha256 {got}, esperado {MODEL_SHA256}). "
                f"Baixe manualmente de {MODEL_URL} e use --face-model."
            )
        os.replace(tmp_path, destination)
        return destination
    except OSError as exc:  # rede fora, DNS, proxy, disco...
        raise ModelUnavailable(
            f"falhou o download do modelo de rostos ({exc}). "
            f"Baixe {MODEL_URL} e passe --face-model CAMINHO, "
            "ou rode com --face-mode off (sem protecao de rosto)."
        ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def resolve_model(explicit: Optional[str] = None, allow_download: bool = True) -> Optional[str]:
    """Devolve o caminho do YuNet, ou ``None`` se nao houver nenhum.

    ``None`` nao e erro aqui: instalacoes com OpenCV 4.x ainda conseguem cair
    no classificador Haar empacotado. Quem decide e o :class:`FaceDetector`.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise ModelUnavailable(f"modelo de rosto nao encontrado: {explicit}")
        return explicit

    from_env = os.environ.get(ENV_MODEL)
    if from_env:
        if not os.path.exists(from_env):
            raise ModelUnavailable(f"{ENV_MODEL} aponta para um arquivo inexistente: {from_env}")
        return from_env

    cached = cached_model_path()
    if os.path.exists(cached):
        return cached

    if allow_download:
        return download_model(cached)
    return None
