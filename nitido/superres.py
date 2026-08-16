"""Super-resolucao por rede neural (Real-ESRGAN compacto).

Interpolacao — Lanczos, bicubica, seja qual for — nao inventa detalhe: ela
so redistribui o que ja esta la. Para *reconstruir* pixel e preciso uma rede
treinada em milhoes de pares (imagem boa, imagem degradada). E o que este
modulo traz: o ``realesr-general-x4v3`` do Real-ESRGAN, uma rede pequena
(~5 MB) com 33 convolucoes.

Rodar isso normalmente exigiria PyTorch, ~2,5 GB de dependencia. Aqui os
pesos sao lidos direto do arquivo ``.pth`` (que e apenas um zip com pickle e
buffers) e montados como um grafo ONNX, executado pelo ONNX Runtime. O
resultado e o mesmo, com uma fracao do peso.

Arquitetura (SRVGGNetCompact):

    conv 3->64 ; PReLU ; [conv 64->64 ; PReLU] x32 ; conv 64->48 ;
    PixelShuffle(4) ; + entrada ampliada por vizinho mais proximo
"""

from __future__ import annotations

import collections
import os
import pickle
import zipfile
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "CharStorage": np.int8,
    "ShortStorage": np.int16,
}


class SuperResUnavailable(RuntimeError):
    """Faltou o modelo, ou o onnxruntime nao esta instalado."""


class _Ignorado:
    """Classes do pickle que nao interessam para a leitura dos pesos."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass


def read_pth(path: str) -> Dict[str, np.ndarray]:
    """Le um ``.pth`` do PyTorch e devolve ``{nome: array numpy}``, sem torch.

    Um ``.pth`` e um zip contendo ``data.pkl`` (o pickle do dicionario de
    pesos) e ``data/<n>`` (os buffers crus). O unpickler abaixo troca as
    classes do torch por equivalentes numpy.
    """
    archive = zipfile.ZipFile(path)
    root = archive.namelist()[0].split("/")[0]

    def rebuild_tensor(storage, offset, size, stride, *_rest):
        count = int(np.prod(size)) if len(size) else 1
        flat = storage[offset : offset + count]
        return np.array(flat, copy=True).reshape(tuple(size))

    class _Reader(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if name in _STORAGE_DTYPES:
                return _STORAGE_DTYPES[name]
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if name == "OrderedDict":
                return collections.OrderedDict
            return _Ignorado

        def persistent_load(self, pid):
            _tag, storage_type, key, _location, _numel = pid
            dtype = storage_type if isinstance(storage_type, type) else np.float32
            return np.frombuffer(archive.read(f"{root}/data/{key}"), dtype=dtype)

    loaded = _Reader(archive.open(f"{root}/data.pkl")).load()
    if isinstance(loaded, dict):
        for chave in ("params", "params_ema", "state_dict"):
            if chave in loaded:
                return loaded[chave]
    return loaded


def build_onnx(weights: Dict[str, np.ndarray], destination: str, scale: int = 4) -> str:
    """Monta o grafo ONNX do SRVGGNetCompact a partir dos pesos e grava."""
    try:
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise SuperResUnavailable(
            "o pacote 'onnx' e necessario para preparar o modelo de IA: "
            "pip install onnx onnxruntime"
        ) from exc

    indices = sorted({int(k.split(".")[1]) for k in weights if k.startswith("body.")})
    if not indices:
        raise SuperResUnavailable("pesos nao parecem ser de um SRVGGNetCompact")

    nodes = []
    initializers = []
    atual = "entrada"

    for i in indices:
        peso = weights[f"body.{i}.weight"]
        if peso.ndim == 4:  # convolucao
            bias = weights[f"body.{i}.bias"]
            nome_w, nome_b = f"w{i}", f"b{i}"
            initializers += [
                numpy_helper.from_array(peso.astype(np.float32), nome_w),
                numpy_helper.from_array(bias.astype(np.float32), nome_b),
            ]
            saida = f"conv{i}"
            nodes.append(
                helper.make_node(
                    "Conv",
                    [atual, nome_w, nome_b],
                    [saida],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                )
            )
        else:  # PReLU, um coeficiente por canal
            nome_s = f"s{i}"
            initializers.append(
                numpy_helper.from_array(
                    peso.astype(np.float32).reshape(-1, 1, 1), nome_s
                )
            )
            saida = f"prelu{i}"
            nodes.append(helper.make_node("PRelu", [atual, nome_s], [saida]))
        atual = saida

    # PixelShuffle do PyTorch == DepthToSpace do ONNX no modo CRD.
    nodes.append(
        helper.make_node("DepthToSpace", [atual], ["shuffled"], blocksize=scale, mode="CRD")
    )

    # Conexao residual: a entrada ampliada por vizinho mais proximo.
    initializers += [
        numpy_helper.from_array(np.array([], np.float32), "roi_vazio"),
        numpy_helper.from_array(
            np.array([1.0, 1.0, float(scale), float(scale)], np.float32), "escalas"
        ),
    ]
    nodes.append(
        helper.make_node(
            "Resize",
            ["entrada", "roi_vazio", "escalas"],
            ["base"],
            mode="nearest",
            nearest_mode="floor",
            coordinate_transformation_mode="asymmetric",
        )
    )
    nodes.append(helper.make_node("Add", ["shuffled", "base"], ["saida"]))

    grafo = helper.make_graph(
        nodes,
        "srvgg_compact",
        [helper.make_tensor_value_info("entrada", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("saida", TensorProto.FLOAT, [1, 3, "H", "W"])],
        initializers,
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 13)])
    modelo.ir_version = 9  # compativel com onnxruntime mais antigos

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(modelo.SerializeToString())
    return destination


class SuperResolver:
    """Executa o modelo de super-resolucao em blocos, com sobreposicao."""

    def __init__(
        self,
        onnx_path: str,
        scale: int = 4,
        tile: int = 256,
        overlap: int = 16,
        threads: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise SuperResUnavailable(
                "o modo de IA precisa do onnxruntime: pip install onnxruntime"
            ) from exc

        opcoes = ort.SessionOptions()
        opcoes.log_severity_level = 3
        if threads > 0:
            opcoes.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opcoes, providers=["CPUExecutionProvider"]
        )
        self.scale = scale
        self.tile = max(64, tile)
        self.overlap = max(4, overlap)

    def _run(self, bgr: np.ndarray) -> np.ndarray:
        rgb = bgr[:, :, ::-1]
        entrada = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        saida = self.session.run(["saida"], {"entrada": entrada})[0]
        return saida[0].transpose(1, 2, 0)[:, :, ::-1]

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        """Amplia ``bgr`` (float32 em ``[0, 1]``) pelo fator do modelo.

        Blocos com sobreposicao mantem a memoria sob controle; so o miolo de
        cada bloco e aproveitado, entao nao aparece emenda.
        """
        h, w = bgr.shape[:2]
        s = self.scale
        out = np.empty((h * s, w * s, 3), np.float32)

        passo = self.tile
        for y0 in range(0, h, passo):
            for x0 in range(0, w, passo):
                y1, x1 = min(h, y0 + passo), min(w, x0 + passo)
                ya, xa = max(0, y0 - self.overlap), max(0, x0 - self.overlap)
                yb, xb = min(h, y1 + self.overlap), min(w, x1 + self.overlap)

                bloco = self._run(np.clip(bgr[ya:yb, xa:xb], 0.0, 1.0))
                topo, esq = (y0 - ya) * s, (x0 - xa) * s
                out[y0 * s : y1 * s, x0 * s : x1 * s] = bloco[
                    topo : topo + (y1 - y0) * s, esq : esq + (x1 - x0) * s
                ]

        return np.clip(out, 0.0, 1.0)


def prepare(pth_path: str, onnx_path: str, scale: int = 4) -> str:
    """Converte o ``.pth`` em ``.onnx`` (uma vez) e devolve o caminho."""
    if os.path.exists(onnx_path):
        return onnx_path
    return build_onnx(read_pth(pth_path), onnx_path, scale=scale)
