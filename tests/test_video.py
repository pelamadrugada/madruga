import cv2
import numpy as np
import pytest

from nitido.pipeline import EnhanceConfig, is_video, process_video_file


def _escreve_video(path, frames=6, size=64, fps=12.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        pytest.skip("codec mp4v indisponivel nesta instalacao do OpenCV")
    rng = np.random.default_rng(5)
    try:
        for i in range(frames):
            frame = (rng.random((size, size, 3)) * 255).astype(np.uint8)
            cv2.rectangle(frame, (5 + i, 5), (30 + i, 40), (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_is_video_por_extensao():
    assert is_video("clipe.mp4")
    assert is_video("CLIPE.MOV")
    assert not is_video("foto.jpg")


def test_process_video_file(tmp_path):
    src = _escreve_video(tmp_path / "entrada.mp4")
    dst = tmp_path / "saida.mp4"

    info = process_video_file(
        str(src), str(dst), EnhanceConfig(scale=2.0, face_mode="off", band_height=32)
    )

    assert info["frames"] == 6
    assert info["output_size"] == (128, 128)
    assert dst.exists() and dst.stat().st_size > 0

    cap = cv2.VideoCapture(str(dst))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame.shape == (128, 128, 3)
    finally:
        cap.release()


def test_max_frames_limita(tmp_path):
    src = _escreve_video(tmp_path / "entrada.mp4", frames=8)
    dst = tmp_path / "saida.mp4"

    info = process_video_file(
        str(src),
        str(dst),
        EnhanceConfig(scale=2.0, face_mode="off", band_height=32),
        max_frames=3,
    )
    assert info["frames"] == 3


def test_progress_e_chamado(tmp_path):
    src = _escreve_video(tmp_path / "entrada.mp4", frames=4)
    vistos = []

    process_video_file(
        str(src),
        str(tmp_path / "saida.mp4"),
        EnhanceConfig(scale=2.0, face_mode="off", band_height=32),
        progress=lambda i, total, report: vistos.append(i),
    )
    assert vistos == [0, 1, 2, 3]


def test_video_inexistente(tmp_path):
    with pytest.raises(IOError):
        process_video_file(str(tmp_path / "nada.mp4"), str(tmp_path / "o.mp4"))


def test_blur_interval_maior_que_um(tmp_path):
    """Com reaproveitamento de PSF o video continua saindo inteiro."""
    src = _escreve_video(tmp_path / "entrada.mp4", frames=6)
    dst = tmp_path / "saida.mp4"

    info = process_video_file(
        str(src),
        str(dst),
        EnhanceConfig(scale=2.0, face_mode="off", band_height=32, blur_interval=3),
    )
    assert info["frames"] == 6
    assert dst.stat().st_size > 0
