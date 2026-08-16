import cv2
import numpy as np
import pytest

from nitido import superres
from nitido.pipeline import EnhanceConfig, build_resolver, enhance_image

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


# --------------------------------------------------------------------------
# Leitura dos pesos sem PyTorch
# --------------------------------------------------------------------------


def test_read_pth_le_pesos_do_modelo_real(modelo_ia):
    pesos = superres.read_pth(modelo_ia)
    assert "body.0.weight" in pesos
    assert pesos["body.0.weight"].shape == (64, 3, 3, 3)
    assert pesos["body.0.weight"].dtype == np.float32
    # ultima convolucao: 3 canais * 4^2 do PixelShuffle
    ultima = max(int(k.split(".")[1]) for k in pesos if k.startswith("body."))
    assert pesos[f"body.{ultima}.weight"].shape[0] == 3 * 4**2


# --------------------------------------------------------------------------
# O grafo ONNX confere com uma implementacao independente em numpy?
# --------------------------------------------------------------------------


def _conv_numpy(x, peso, bias):
    """Convolucao 3x3 'same' em NumPy — referencia lenta e obvia."""
    saida = np.zeros((peso.shape[0], x.shape[1], x.shape[2]), np.float32)
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    for o in range(peso.shape[0]):
        acc = np.zeros(x.shape[1:], np.float32)
        for i in range(peso.shape[1]):
            for ky in range(3):
                for kx in range(3):
                    acc += peso[o, i, ky, kx] * padded[i, ky : ky + x.shape[1], kx : kx + x.shape[2]]
        saida[o] = acc + bias[o]
    return saida


def _pixel_shuffle_numpy(x, escala):
    c, h, w = x.shape
    saida = x.reshape(c // escala**2, escala, escala, h, w)
    saida = saida.transpose(0, 3, 1, 4, 2)
    return saida.reshape(c // escala**2, h * escala, w * escala)


def _referencia(pesos, imagem, escala):
    """SRVGGNetCompact inteiro em numpy, para comparar com o ONNX."""
    x = imagem[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # BGR->RGB, HWC->CHW
    atual = x
    for i in sorted({int(k.split(".")[1]) for k in pesos if k.startswith("body.")}):
        peso = pesos[f"body.{i}.weight"]
        if peso.ndim == 4:
            atual = _conv_numpy(atual, peso, pesos[f"body.{i}.bias"])
        else:  # PReLU
            coef = peso.reshape(-1, 1, 1)
            atual = np.where(atual >= 0, atual, coef * atual)
    atual = _pixel_shuffle_numpy(atual, escala)
    base = np.repeat(np.repeat(x, escala, axis=1), escala, axis=2)
    return (atual + base).transpose(1, 2, 0)[:, :, ::-1]


def _pesos_sinteticos(num_conv=2, feat=8, escala=2, seed=0):
    rng = np.random.default_rng(seed)
    pesos = {}
    pesos["body.0.weight"] = rng.normal(0, 0.2, (feat, 3, 3, 3)).astype(np.float32)
    pesos["body.0.bias"] = rng.normal(0, 0.1, feat).astype(np.float32)
    pesos["body.1.weight"] = rng.uniform(0, 0.3, feat).astype(np.float32)
    idx = 2
    for _ in range(num_conv):
        pesos[f"body.{idx}.weight"] = rng.normal(0, 0.2, (feat, feat, 3, 3)).astype(np.float32)
        pesos[f"body.{idx}.bias"] = rng.normal(0, 0.1, feat).astype(np.float32)
        pesos[f"body.{idx + 1}.weight"] = rng.uniform(0, 0.3, feat).astype(np.float32)
        idx += 2
    pesos[f"body.{idx}.weight"] = rng.normal(0, 0.2, (3 * escala**2, feat, 3, 3)).astype(np.float32)
    pesos[f"body.{idx}.bias"] = rng.normal(0, 0.1, 3 * escala**2).astype(np.float32)
    return pesos


def test_grafo_onnx_bate_com_a_referencia_em_numpy(tmp_path):
    """Valida Conv, PReLU, PixelShuffle (DepthToSpace CRD) e o residual."""
    escala = 2
    pesos = _pesos_sinteticos(escala=escala)
    onnx_path = superres.build_onnx(pesos, str(tmp_path / "m.onnx"), scale=escala)

    rng = np.random.default_rng(3)
    imagem = rng.random((12, 10, 3)).astype(np.float32)

    resolver = superres.SuperResolver(onnx_path, scale=escala, tile=256)
    obtido = resolver._run(imagem)
    esperado = _referencia(pesos, imagem, escala)

    assert obtido.shape == (24, 20, 3)
    np.testing.assert_allclose(obtido, esperado, atol=2e-4)


def test_blocos_dao_o_mesmo_resultado_que_a_imagem_inteira(tmp_path):
    """Sobreposicao maior que o campo receptivo elimina emenda entre blocos."""
    escala = 2
    pesos = _pesos_sinteticos(num_conv=2, escala=escala)
    onnx_path = superres.build_onnx(pesos, str(tmp_path / "m.onnx"), scale=escala)

    rng = np.random.default_rng(4)
    imagem = rng.random((96, 96, 3)).astype(np.float32)

    inteiro = superres.SuperResolver(onnx_path, scale=escala, tile=4096).upscale(imagem)
    blocos = superres.SuperResolver(onnx_path, scale=escala, tile=32, overlap=12).upscale(imagem)

    np.testing.assert_allclose(inteiro, blocos, atol=1e-5)


def test_upscale_respeita_o_fator(tmp_path):
    pesos = _pesos_sinteticos(escala=2)
    onnx_path = superres.build_onnx(pesos, str(tmp_path / "m.onnx"), scale=2)
    resolver = superres.SuperResolver(onnx_path, scale=2, tile=64)
    saida = resolver.upscale(np.zeros((40, 30, 3), np.float32))
    assert saida.shape == (80, 60, 3)
    assert saida.min() >= 0.0 and saida.max() <= 1.0


def test_pesos_invalidos_reclamam(tmp_path):
    with pytest.raises(superres.SuperResUnavailable):
        superres.build_onnx({"outra.coisa": np.zeros((1,), np.float32)}, str(tmp_path / "m.onnx"))


# --------------------------------------------------------------------------
# Integracao com o pipeline
# --------------------------------------------------------------------------


def test_engine_auto_cai_no_classico_sem_modelo():
    """Sem modelo, 'auto' nao pode explodir — so deixa de usar a IA."""
    cfg = EnhanceConfig(engine="auto", ai_allow_download=False)
    assert build_resolver(cfg) is None


def test_engine_ai_sem_modelo_e_erro():
    """Quem pediu IA explicitamente precisa saber que ela nao rodou."""
    cfg = EnhanceConfig(engine="ai", ai_allow_download=False)
    with pytest.raises(RuntimeError):
        build_resolver(cfg)


def test_engine_classic_nunca_cria_resolvedor():
    assert build_resolver(EnhanceConfig(engine="classic")) is None


def test_relatorio_diz_qual_motor_rodou():
    imagem = (np.random.default_rng(1).random((40, 40, 3)) * 255).astype(np.uint8)
    _, report = enhance_image(imagem, EnhanceConfig(scale=2.0, engine="classic", face_mode="off"))
    assert report.engine == "classic"


def test_pipeline_com_ia_de_verdade(modelo_ia, tmp_path):
    """Ponta a ponta com o modelo real: o rosto continua protegido."""
    from nitido.enhance import to_uint8, upscale
    from nitido.faces import FaceBox

    rng = np.random.default_rng(9)
    base = rng.random((20, 20, 3)).astype(np.float32)
    imagem = to_uint8(cv2.resize(base, (200, 200), interpolation=cv2.INTER_CUBIC))
    rosto = FaceBox(60, 60, 80, 80)

    cfg = EnhanceConfig(
        scale=4.0, engine="ai", ai_model=modelo_ia, band_height=128, clahe_clip=0, sharpen_amount=0
    )
    saida, report = enhance_image(imagem, cfg, boxes=[rosto])

    assert report.engine == "ai"
    assert saida.shape == (800, 800, 3)

    # Nucleo do rosto: identico ao Lanczos puro, a IA nao entrou ali.
    referencia = to_uint8(upscale(imagem.astype(np.float32) / 255.0, 4.0))
    y0, y1 = 95 * 4, 115 * 4
    diff = np.abs(saida[y0:y1, y0:y1].astype(int) - referencia[y0:y1, y0:y1].astype(int))
    assert diff.max() <= 2, f"a IA alterou o rosto (dif max {diff.max()})"

    # Fora do rosto: a IA precisa ter mudado alguma coisa de verdade.
    fundo = np.abs(saida[:200, :200].astype(int) - referencia[:200, :200].astype(int))
    assert fundo.mean() > 1.0
