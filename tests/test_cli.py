import cv2
import numpy as np
import pytest

from nitido.cli import _default_output, build_parser, config_from_args, main


def _imagem(size=80):
    rng = np.random.default_rng(11)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_default_output_nomeia_com_a_escala(tmp_path):
    src = tmp_path / "foto.jpg"
    assert _default_output(str(src), 5.0).endswith("foto_5x.jpg")
    assert _default_output(str(src), 2.5).endswith("foto_2_5x.jpg")


def test_default_output_respeita_pasta(tmp_path):
    saida = _default_output(str(tmp_path / "foto.png"), 5.0, out_dir=str(tmp_path / "out"))
    assert saida == str(tmp_path / "out" / "foto_5x.png")


def test_config_from_args_le_as_flags():
    args = build_parser().parse_args(
        ["entrada.jpg", "--scale", "3", "--deblur", "rl", "--face-mode", "gentle", "--sharpen", "0.4"]
    )
    cfg = config_from_args(args)
    assert cfg.scale == 3.0
    assert cfg.deblur_method == "rl"
    assert cfg.face_mode == "gentle"
    assert cfg.sharpen_amount == 0.4


def test_cli_processa_uma_imagem(tmp_path, capsys):
    src = tmp_path / "foto.png"
    dst = tmp_path / "grande.png"
    cv2.imwrite(str(src), _imagem())

    code = main([str(src), "-o", str(dst), "--scale", "2", "--face-mode", "off"])

    assert code == 0
    assert cv2.imread(str(dst)).shape == (160, 160, 3)
    assert "[imagem]" in capsys.readouterr().out


def test_cli_gera_nome_padrao(tmp_path):
    src = tmp_path / "foto.png"
    cv2.imwrite(str(src), _imagem())

    assert main([str(src), "--scale", "2", "--face-mode", "off", "-q"]) == 0
    assert (tmp_path / "foto_2x.png").exists()


def test_cli_lote_em_pasta(tmp_path):
    entrada = tmp_path / "in"
    saida = tmp_path / "out"
    entrada.mkdir()
    for nome in ("a.png", "b.png"):
        cv2.imwrite(str(entrada / nome), _imagem())
    (entrada / "leiame.txt").write_text("ignorar")

    code = main([str(entrada), "-o", str(saida), "--scale", "2", "--face-mode", "off", "-q"])

    assert code == 0
    assert (saida / "a_2x.png").exists()
    assert (saida / "b_2x.png").exists()
    assert not (saida / "leiame_2x.txt").exists()


def test_cli_recusa_sobrescrever_a_entrada(tmp_path, capsys):
    src = tmp_path / "foto.png"
    cv2.imwrite(str(src), _imagem())

    code = main([str(src), "-o", str(src), "--face-mode", "off", "-q"])

    assert code == 1
    assert "saida igual a entrada" in capsys.readouterr().err


def test_cli_entrada_inexistente(tmp_path, capsys):
    code = main([str(tmp_path / "nada.png")])
    assert code == 2
    assert "erro:" in capsys.readouterr().err


def test_cli_scale_invalida(tmp_path, capsys):
    src = tmp_path / "foto.png"
    cv2.imwrite(str(src), _imagem())
    assert main([str(src), "--scale", "0"]) == 2
    assert "erro:" in capsys.readouterr().err


def test_cli_erro_por_estouro_de_memoria(tmp_path, capsys):
    src = tmp_path / "foto.png"
    cv2.imwrite(str(src), _imagem())
    code = main(
        [str(src), "-o", str(tmp_path / "o.png"), "--max-pixels", "0.001", "--face-mode", "off", "-q"]
    )
    assert code == 1
    assert "acima do limite" in capsys.readouterr().err


def test_help_nao_quebra(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "motion blur" in capsys.readouterr().out
