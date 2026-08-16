import os

import pytest

from nitido import models


@pytest.fixture(autouse=True)
def cache_isolado(tmp_path_factory, monkeypatch):
    """Testes nunca tocam no cache real nem na rede sem pedir.

    O cache aponta para um diretorio temporario e ``download_model`` levanta
    :class:`ModelUnavailable`. Quem precisa de download de verdade sobrescreve
    o comportamento explicitamente.
    """
    monkeypatch.setenv(models.ENV_CACHE, str(tmp_path_factory.mktemp("cache")))
    monkeypatch.delenv(models.ENV_MODEL, raising=False)

    def sem_rede(*_args, **_kwargs):
        raise models.ModelUnavailable("download desabilitado durante os testes")

    monkeypatch.setattr(models, "download_model", sem_rede)
    yield


@pytest.fixture
def modelo_yunet():
    """Caminho de um YuNet real, se o ambiente tiver um; senao pula o teste."""
    path = os.environ.get("NITIDO_TEST_FACE_MODEL")
    if not path or not os.path.exists(path):
        pytest.skip("defina NITIDO_TEST_FACE_MODEL para rodar os testes de deteccao")
    return path
