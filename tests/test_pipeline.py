import cv2
import numpy as np
import pytest

from nitido import deblur as db
from nitido.enhance import to_uint8, upscale
from nitido.faces import FaceBox
from nitido.pipeline import EnhanceConfig, enhance_image, process_image_file

FACE = FaceBox(60, 60, 80, 80)


def _scene(size=200, seed=7):
    """Cena sintetica: textura no fundo e um 'rosto' liso no meio."""
    rng = np.random.default_rng(seed)
    base = rng.random((size // 10, size // 10, 3)).astype(np.float32)
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(img, (10, 10), (55, 190), (1.0, 1.0, 1.0), thickness=2)
    cv2.circle(img, (FACE.x + 40, FACE.y + 40), 35, (0.75, 0.7, 0.68), thickness=-1)
    return to_uint8(np.clip(img, 0, 1))


def _cfg(**kw):
    base = dict(scale=5.0, band_height=64)
    base.update(kw)
    return EnhanceConfig(**base)


def test_saida_tem_exatamente_5x():
    out, report = enhance_image(_scene(), _cfg(), boxes=[FACE])
    assert out.shape == (1000, 1000, 3)
    assert out.dtype == np.uint8
    assert report.output_size == (1000, 1000)
    assert report.input_size == (200, 200)
    assert report.faces == 1


def test_escala_diferente_de_5_tambem_funciona():
    out, _ = enhance_image(_scene(), _cfg(scale=2.5), boxes=[FACE])
    assert out.shape == (500, 500, 3)


def test_rosto_fica_igual_ao_simples_redimensionamento():
    """O nucleo do rosto tem que ser so o Lanczos do original — nada mais."""
    src = _scene()
    out, _ = enhance_image(src, _cfg(), boxes=[FACE])
    referencia = to_uint8(upscale(src.astype(np.float32) / 255.0, 5.0))

    # Nucleo do rosto, bem dentro da mascara (a borda e uma transicao suave).
    y0, y1 = int(95 * 5), int(115 * 5)
    x0, x1 = int(95 * 5), int(115 * 5)
    diff = np.abs(out[y0:y1, x0:x1].astype(int) - referencia[y0:y1, x0:x1].astype(int))
    assert diff.max() <= 2, f"rosto foi alterado (dif max {diff.max()})"


def test_fundo_ganha_detalhe():
    src = _scene()
    out, _ = enhance_image(src, _cfg(), boxes=[FACE])
    referencia = to_uint8(upscale(src.astype(np.float32) / 255.0, 5.0))

    def nitidez(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Faixa da esquerda: longe do rosto, so fundo.
    assert nitidez(out[:, :250]) > nitidez(referencia[:, :250]) * 1.2


def test_face_mode_enhance_alcanca_o_rosto():
    src = _scene()
    protegido, _ = enhance_image(src, _cfg(face_mode="preserve"), boxes=[FACE])
    realcado, _ = enhance_image(src, _cfg(face_mode="enhance"), boxes=[FACE])
    y0, y1, x0, x1 = 475, 575, 475, 575
    assert not np.array_equal(protegido[y0:y1, x0:x1], realcado[y0:y1, x0:x1])


def test_face_mode_off_nao_usa_detector():
    out, report = enhance_image(_scene(), _cfg(face_mode="off"), boxes=[FACE])
    assert report.faces == 0
    assert out.shape == (1000, 1000, 3)


def test_faixas_menores_dao_o_mesmo_resultado():
    """O processamento em faixas nao pode deixar emenda visivel."""
    src = _scene()
    inteiro, _ = enhance_image(src, _cfg(band_height=4096), boxes=[FACE])
    faixas, _ = enhance_image(src, _cfg(band_height=32), boxes=[FACE])
    diff = np.abs(inteiro.astype(int) - faixas.astype(int))
    assert diff.max() <= 2, f"emenda entre faixas (dif max {diff.max()})"


def test_deblur_e_aplicado_em_imagem_borrada():
    src = _scene()
    psf = db.motion_psf(13, 0.0)
    borrada = to_uint8(
        cv2.filter2D(src.astype(np.float32) / 255.0, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    )
    _, report = enhance_image(borrada, _cfg(), boxes=[FACE])
    assert report.deblur_applied is True
    assert report.blur_length == pytest.approx(13, abs=4)


def test_rosto_continua_intocado_mesmo_com_deblur():
    src = _scene()
    psf = db.motion_psf(13, 0.0)
    borrada = to_uint8(
        cv2.filter2D(src.astype(np.float32) / 255.0, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    )
    out, report = enhance_image(borrada, _cfg(), boxes=[FACE])
    assert report.deblur_applied is True

    referencia = to_uint8(upscale(borrada.astype(np.float32) / 255.0, 5.0))
    y0, y1, x0, x1 = 475, 575, 475, 575
    diff = np.abs(out[y0:y1, x0:x1].astype(int) - referencia[y0:y1, x0:x1].astype(int))
    assert diff.max() <= 2


def test_entrada_em_tons_de_cinza_vira_bgr():
    gray = cv2.cvtColor(_scene(), cv2.COLOR_BGR2GRAY)
    out, _ = enhance_image(gray, _cfg(scale=2.0), boxes=[])
    assert out.shape == (400, 400, 3)


def test_entrada_com_alpha_e_aceita():
    bgra = cv2.cvtColor(_scene(), cv2.COLOR_BGR2BGRA)
    out, _ = enhance_image(bgra, _cfg(scale=2.0), boxes=[])
    assert out.shape == (400, 400, 3)


def test_trava_de_memoria():
    with pytest.raises(MemoryError):
        enhance_image(_scene(), _cfg(max_output_pixels=1000), boxes=[FACE])


@pytest.mark.parametrize(
    "kw",
    [
        {"scale": 0},
        {"face_mode": "bonito"},
        {"deblur_method": "magia"},
        {"interpolation": "quantica"},
        {"band_height": 4},
    ],
)
def test_config_invalida_levanta(kw):
    with pytest.raises(ValueError):
        _cfg(**kw).validate()


def test_process_image_file(tmp_path):
    src = tmp_path / "entrada.png"
    dst = tmp_path / "sub" / "saida.png"
    cv2.imwrite(str(src), _scene())

    report = process_image_file(str(src), str(dst), _cfg(scale=2.0, face_mode="off"))

    assert dst.exists()
    gravada = cv2.imread(str(dst))
    assert gravada.shape == (400, 400, 3)
    assert report.output_size == (400, 400)


def test_process_image_file_arquivo_inexistente(tmp_path):
    with pytest.raises(IOError):
        process_image_file(str(tmp_path / "nao_existe.png"), str(tmp_path / "o.png"), _cfg())
