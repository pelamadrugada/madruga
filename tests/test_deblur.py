import cv2
import numpy as np
import pytest

from nitido import deblur as db


def _texture(size=256, seed=0):
    """Imagem sintetica nitida: detalhe fino em todas as direcoes + bordas."""
    rng = np.random.default_rng(seed)
    img = rng.random((size, size)).astype(np.float32)
    # Um respiro de 0.6 px tira o aliasing sem tirar a nitidez — e o que uma
    # foto bem focada parece no espectro.
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    cv2.rectangle(img, (40, 40), (size - 60, size - 40), 1.0, thickness=3)
    cv2.circle(img, (size - 70, 70), 30, 0.0, thickness=2)
    return np.clip(img, 0.0, 1.0)


def test_motion_psf_normalizado():
    psf = db.motion_psf(9, 0.0)
    assert psf.shape == (9, 9)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)


def test_motion_psf_horizontal_e_uma_linha():
    psf = db.motion_psf(11, 0.0)
    linha = psf[psf.shape[0] // 2]
    assert linha.sum() == pytest.approx(1.0, abs=1e-6)
    assert psf.sum(axis=1).argmax() == psf.shape[0] // 2


def test_motion_psf_vertical_e_uma_coluna():
    psf = db.motion_psf(11, 90.0)
    assert psf.sum(axis=0).argmax() == psf.shape[1] // 2
    assert psf[:, psf.shape[1] // 2].sum() == pytest.approx(1.0, abs=1e-6)


def test_psf_de_comprimento_1_e_identidade():
    psf = db.motion_psf(1, 37.0)
    assert psf.shape == (1, 1)
    assert psf[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("angulo", [0.0, 45.0, 90.0, 135.0])
def test_estimativa_recupera_angulo_e_comprimento(angulo):
    sharp = _texture(256)
    psf = db.motion_psf(15, angulo)
    blurred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)

    length, ang, conf = db.estimate_motion_psf(blurred, max_length=40)

    assert conf > 8.0, "borrao evidente deveria dar confianca bem acima do limiar"
    assert abs(length - 15) <= 3
    erro = min(abs(ang - angulo), 180 - abs(ang - angulo))
    assert erro <= 12


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_imagem_nitida_nao_dispara_deblur(seed):
    """Falso positivo aqui significaria criar anel numa foto que estava boa."""
    sharp = _texture(256, seed=seed)
    bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    out, est = db.deblur(bgr, method="wiener")
    assert est.used is False, f"confianca {est.confidence:.1f} passou do limiar"
    np.testing.assert_array_equal(out, bgr)


def test_desfoque_gaussiano_nao_e_confundido_com_motion_blur():
    """Fora de foco nao e borrao de movimento: nao ha direcao para desfazer."""
    sharp = _texture(256, seed=5)
    fora_de_foco = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    _, est = db.deblur(cv2.cvtColor(fora_de_foco, cv2.COLOR_GRAY2BGR), method="wiener")
    assert est.used is False


@pytest.mark.parametrize("comprimento", [5, 9, 15, 25])
def test_motion_blur_real_dispara_deblur(comprimento):
    sharp = _texture(256, seed=6)
    psf = db.motion_psf(comprimento, 30.0)
    borrada = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    _, est = db.deblur(cv2.cvtColor(borrada, cv2.COLOR_GRAY2BGR), method="wiener")
    assert est.used is True
    assert est.length == pytest.approx(comprimento, abs=2)


def test_wiener_reduz_o_erro():
    sharp = _texture(256, seed=1)
    psf = db.motion_psf(13, 20.0)
    blurred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)

    restored = db.wiener_deconvolve(blurred, psf, noise=0.005)

    def erro(a):
        return float(np.mean((a[20:-20, 20:-20] - sharp[20:-20, 20:-20]) ** 2))

    assert erro(restored) < erro(blurred) * 0.6


def test_richardson_lucy_reduz_o_erro():
    sharp = _texture(192, seed=2)
    psf = db.motion_psf(9, 0.0)
    blurred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)

    restored = db.richardson_lucy(blurred, psf, iterations=20)

    def erro(a):
        return float(np.mean((a[20:-20, 20:-20] - sharp[20:-20, 20:-20]) ** 2))

    assert erro(restored) < erro(blurred) * 0.7


def test_deblur_completo_melhora_nitidez():
    sharp = _texture(256, seed=4)
    psf = db.motion_psf(15, 0.0)
    blurred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    out, est = db.deblur(bgr, method="wiener", strength=1.0)

    assert est.used is True
    def nitidez(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())

    assert nitidez(out) > nitidez(bgr) * 1.5


def test_deblur_none_devolve_entrada():
    bgr = cv2.cvtColor(_texture(96), cv2.COLOR_GRAY2BGR)
    out, est = db.deblur(bgr, method="none")
    assert est.used is False
    np.testing.assert_array_equal(out, bgr)


def test_metodo_invalido_levanta():
    bgr = cv2.cvtColor(_texture(96), cv2.COLOR_GRAY2BGR)
    with pytest.raises(ValueError):
        db.deblur(bgr, method="magia", psf=db.motion_psf(5, 0.0))


def test_psf_pronta_dispensa_estimativa():
    """Video reaproveita a PSF entre quadros; o relatorio tem que segui-la."""
    sharp = _texture(192, seed=8)
    psf = db.motion_psf(11, 20.0)
    borrada = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    bgr = cv2.cvtColor(borrada, cv2.COLOR_GRAY2BGR)

    _, est = db.deblur(bgr, psf=psf, estimate=(11.0, 20.0, 30.0))

    assert est.used is True
    assert est.length == 11.0
    assert est.angle == 20.0
    assert est.psf is psf


def test_strength_zero_nao_altera_nada():
    bgr = cv2.cvtColor(_texture(96), cv2.COLOR_GRAY2BGR)
    out, est = db.deblur(bgr, strength=0.0)
    assert est.used is False
    np.testing.assert_array_equal(out, bgr)
