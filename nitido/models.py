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

# Modelo de super-resolucao: Real-ESRGAN compacto ("general x4 v3"), ~4,9 MB.
# Vem como .pth do PyTorch, mas nitido le os pesos sem torch e converte para
# ONNX no primeiro uso (veja nitido/superres.py).
SR_FILENAME = "realesr-general-x4v3.pth"
SR_ONNX_FILENAME = "realesr-general-x4v3.onnx"
SR_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)
SR_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
SR_SCALE = 4

ENV_SR_MODEL = "NITIDO_SR_MODEL"


class ModelUnavailable(RuntimeError):
    """Nao foi possivel obter um modelo necessario."""


def cache_dir() -> str:
    override = os.environ.get(ENV_CACHE)
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "nitido")


def cached_model_path() -> str:
    return os.path.join(cache_dir(), MODEL_FILENAME)


def cached_superres_path() -> str:
    return os.path.join(cache_dir(), SR_FILENAME)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _baixar(url: str, esperado: str, destination: str, saida: str, timeout: float) -> str:
    """Baixa ``url`` para ``destination`` conferindo o sha256.

    ``saida`` e a instrucao mostrada ao usuario quando o download falha —
    cada modelo tem uma alternativa diferente.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(destination)))
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with open(tmp_path, "wb") as handle:
                while True:
                    block = response.read(1 << 16)
                    if not block:
                        break
                    handle.write(block)

        got = sha256(tmp_path)
        if got != esperado:
            raise ModelUnavailable(
                f"o modelo baixado nao confere (sha256 {got}, esperado {esperado}). "
                f"Baixe manualmente de {url}."
            )
        os.replace(tmp_path, destination)
        return destination
    except OSError as exc:  # rede fora, DNS, proxy, disco...
        raise ModelUnavailable(f"falhou o download ({exc}). {saida}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_model(destination: Optional[str] = None, timeout: float = 60.0) -> str:
    """Baixa o YuNet (deteccao de rostos) para o cache. Devolve o caminho."""
    return _baixar(
        MODEL_URL,
        MODEL_SHA256,
        destination or cached_model_path(),
        f"Baixe {MODEL_URL} e passe --face-model CAMINHO, "
        "ou rode com --face-mode off (sem protecao de rosto).",
        timeout,
    )


def download_superres(destination: Optional[str] = None, timeout: float = 120.0) -> str:
    """Baixa o Real-ESRGAN compacto para o cache. Devolve o caminho do .pth."""
    return _baixar(
        SR_URL,
        SR_SHA256,
        destination or cached_superres_path(),
        f"Baixe {SR_URL} e passe --ai-model CAMINHO, "
        "ou rode com --engine classic (ampliacao sem IA).",
        timeout,
    )


def resolve_superres_model(
    explicit: Optional[str] = None, allow_download: bool = True
) -> Optional[str]:
    """Devolve o caminho do ``.onnx`` pronto para uso, ou ``None``.

    Aceita tanto um ``.onnx`` ja convertido quanto o ``.pth`` original — no
    segundo caso a conversao acontece aqui, uma unica vez, e o resultado fica
    no cache ao lado.
    """
    from .superres import prepare  # importacao tardia: so o modo de IA precisa

    origem = explicit or os.environ.get(ENV_SR_MODEL)
    if origem:
        if not os.path.exists(origem):
            raise ModelUnavailable(f"modelo de IA nao encontrado: {origem}")
        if origem.endswith(".onnx"):
            return origem
        return prepare(origem, os.path.join(cache_dir(), SR_ONNX_FILENAME), SR_SCALE)

    onnx_cache = os.path.join(cache_dir(), SR_ONNX_FILENAME)
    if os.path.exists(onnx_cache):
        return onnx_cache

    pth_cache = cached_superres_path()
    if not os.path.exists(pth_cache):
        if not allow_download:
            return None
        download_superres(pth_cache)
    return prepare(pth_cache, onnx_cache, SR_SCALE)


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
