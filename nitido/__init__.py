"""nitido — ampliacao 5x com remocao de motion blur, preservando o rosto.

O pacote expoe as pecas usadas pela CLI:

* :mod:`nitido.faces`    — deteccao de rostos e mascara de protecao
* :mod:`nitido.deblur`   — estimativa do borrao de movimento e deconvolucao
* :mod:`nitido.enhance`  — contraste local, unsharp e redimensionamento
* :mod:`nitido.pipeline` — orquestracao para imagem e video
"""

from .pipeline import EnhanceConfig, enhance_image, process_image_file, process_video_file

__version__ = "0.1.0"

__all__ = [
    "EnhanceConfig",
    "enhance_image",
    "process_image_file",
    "process_video_file",
    "__version__",
]
