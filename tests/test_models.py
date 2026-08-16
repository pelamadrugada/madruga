import os

import cv2
import numpy as np
import pytest

from nitido import models
from nitido.faces import FaceDetector
from nitido.models import ModelUnavailable, resolve_model


def test_cache_dir_respeita_a_variavel(monkeypatch, tmp_path):
    monkeypatch.setenv(models.ENV_CACHE, str(tmp_path))
    assert models.cache_dir() == str(tmp_path)
    assert models.cached_model_path() == str(tmp_path / models.MODEL_FILENAME)


def test_resolve_usa_o_caminho_explicito(tmp_path):
    arquivo = tmp_path / "meu.onnx"
    arquivo.write_bytes(b"conteudo")
    assert resolve_model(str(arquivo)) == str(arquivo)


def test_resolve_reclama_de_caminho_explicito_inexistente(tmp_path):
    with pytest.raises(ModelUnavailable):
        resolve_model(str(tmp_path / "nao_existe.onnx"))


def test_resolve_usa_a_variavel_de_ambiente(monkeypatch, tmp_path):
    arquivo = tmp_path / "env.onnx"
    arquivo.write_bytes(b"conteudo")
    monkeypatch.setenv(models.ENV_MODEL, str(arquivo))
    assert resolve_model() == str(arquivo)


def test_resolve_reclama_de_variavel_apontando_para_o_nada(monkeypatch, tmp_path):
    monkeypatch.setenv(models.ENV_MODEL, str(tmp_path / "fantasma.onnx"))
    with pytest.raises(ModelUnavailable):
        resolve_model()


def test_resolve_acha_o_arquivo_em_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(models.ENV_CACHE, str(tmp_path))
    cached = tmp_path / models.MODEL_FILENAME
    cached.write_bytes(b"conteudo")
    assert resolve_model() == str(cached)


def test_resolve_sem_download_devolve_none(monkeypatch, tmp_path):
    monkeypatch.setenv(models.ENV_CACHE, str(tmp_path))
    assert resolve_model(allow_download=False) is None


def test_sha256(tmp_path):
    arquivo = tmp_path / "x.bin"
    arquivo.write_bytes(b"nitido")
    assert models.sha256(str(arquivo)) == (
        "0cbef0018dc8b5919a64ea33da784567fffdabb936dbbb87f3e1cc153f40a4f1"
    )


def test_detector_sem_modelo_e_sem_haar_explica_o_problema(monkeypatch):
    monkeypatch.setattr(FaceDetector, "_load_cascades", lambda self: None)
    with pytest.raises(ModelUnavailable) as exc:
        FaceDetector(allow_download=False)
    assert "--face-mode off" in str(exc.value)


@pytest.mark.skipif(
    not hasattr(cv2, "FaceDetectorYN"), reason="OpenCV sem FaceDetectorYN"
)
def test_detector_yunet_roda(modelo_yunet):
    detector = FaceDetector(model=modelo_yunet)
    assert detector.backend == "yunet"
    ruido = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    assert detector.detect(ruido) == []  # ruido nao tem rosto


def test_detector_yunet_encontra_rosto_de_verdade(modelo_yunet):
    """So roda quando ha uma foto de referencia com rosto no ambiente."""
    foto = os.environ.get("NITIDO_TEST_FACE_IMAGE")
    if not foto or not os.path.exists(foto):
        pytest.skip("defina NITIDO_TEST_FACE_IMAGE com uma foto contendo rosto")
    imagem = cv2.imread(foto, cv2.IMREAD_COLOR)
    rostos = FaceDetector(model=modelo_yunet).detect(imagem)
    assert len(rostos) >= 1
