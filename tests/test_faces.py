import numpy as np

from nitido.faces import FaceBox, face_mask, _iou, _merge_overlapping


def test_mascara_vazia_sem_rostos():
    mask = face_mask((64, 48), [])
    assert mask.shape == (64, 48)
    assert mask.dtype == np.float32
    assert mask.max() == 0.0


def test_mascara_cobre_o_rosto_e_zera_longe():
    mask = face_mask((400, 400), [FaceBox(150, 150, 100, 100)], expand=0.3, feather=0.3)

    assert mask.shape == (400, 400)
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    assert mask[200, 200] > 0.95, "o centro do rosto tem que estar protegido"
    assert mask[10, 10] < 0.01, "canto distante nao pode ser afetado"


def test_borda_da_mascara_e_suave():
    mask = face_mask((400, 400), [FaceBox(150, 150, 100, 100)], feather=0.4)
    linha = mask[200]
    # Uma transicao suave passa por valores intermediarios; um recorte duro nao.
    assert np.any((linha > 0.2) & (linha < 0.8))


def test_varios_rostos_entram_na_mesma_mascara():
    boxes = [FaceBox(40, 40, 60, 60), FaceBox(300, 300, 60, 60)]
    mask = face_mask((400, 400), boxes)
    assert mask[70, 70] > 0.9
    assert mask[330, 330] > 0.9
    assert mask[200, 200] < 0.1


def test_iou_de_caixas_identicas_e_um():
    box = FaceBox(0, 0, 10, 10)
    assert _iou(box, box) == 1.0


def test_iou_de_caixas_disjuntas_e_zero():
    assert _iou(FaceBox(0, 0, 10, 10), FaceBox(50, 50, 10, 10)) == 0.0


def test_merge_descarta_duplicata_sobreposta():
    boxes = [FaceBox(10, 10, 100, 100), FaceBox(14, 12, 98, 102), FaceBox(300, 300, 50, 50)]
    kept = _merge_overlapping(boxes)
    assert len(kept) == 2


def test_face_box_scaled():
    box = FaceBox(10, 20, 30, 40).scaled(2.0)
    assert (box.x, box.y, box.w, box.h) == (20, 40, 60, 80)
