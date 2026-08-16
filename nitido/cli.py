"""Interface de linha de comando do nitido."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import cv2

from . import __version__
from .pipeline import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    EnhanceConfig,
    Report,
    build_detector,
    build_resolver,
    is_video,
    process_image_file,
    process_video_file,
)

EPILOG = """\
exemplos:
  nitido foto.jpg                          # grava foto_5x.jpg
  nitido foto.jpg -o grande.png            # escolhe o arquivo de saida
  nitido entrada/ -o saida/                # lote: pasta inteira
  nitido clipe.mp4 -o clipe_5x.mp4         # video (sem audio, veja o README)
  nitido foto.jpg --deblur rl --sharpen 0.9
  nitido foto.jpg --face-mode gentle       # deixa o deblur alcancar o rosto
  nitido foto.jpg --face-mode off          # sem nenhuma protecao de rosto
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nitido",
        description=(
            "Amplia 5x, tira motion blur e aumenta o detalhe ao redor — "
            "sem alterar o rosto."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="imagem, video ou pasta de entrada")
    p.add_argument("-o", "--output", help="arquivo ou pasta de saida")
    p.add_argument("--version", action="version", version=f"nitido {__version__}")

    g = p.add_argument_group("motor")
    g.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "ai", "classic"),
        help=(
            "ai: reconstroi pixel com rede neural (Real-ESRGAN); "
            "classic: so interpola, sem IA; "
            "auto (padrao): usa a IA quando disponivel, senao avisa e cai no classic"
        ),
    )
    g.add_argument("--ai-model", help="caminho de um realesr-general-x4v3 (.pth ou .onnx)")
    g.add_argument("--ai-tile", type=int, default=384, help="lado do bloco processado pela rede")
    g.add_argument("--ai-overlap", type=int, default=40, help="sobreposicao entre blocos da rede")
    g.add_argument("--ai-threads", type=int, default=0, help="threads da rede (0 = automatico)")

    g = p.add_argument_group("ampliacao")
    g.add_argument("-s", "--scale", type=float, default=5.0, help="fator de ampliacao (padrao: 5)")
    g.add_argument(
        "--interp",
        default="lanczos",
        choices=("lanczos", "cubic", "linear", "nearest"),
        help="interpolacao da ampliacao (padrao: lanczos)",
    )

    g = p.add_argument_group("motion blur")
    g.add_argument(
        "--deblur",
        default="wiener",
        choices=("wiener", "rl", "none"),
        help="wiener (rapido), rl (Richardson-Lucy, mais limpo) ou none",
    )
    g.add_argument(
        "--deblur-strength",
        type=float,
        default=0.85,
        help="0 a 1: quanto do resultado deconvoluido entra (padrao: 0.85)",
    )
    g.add_argument(
        "--deblur-noise",
        type=float,
        default=0.012,
        help="razao ruido/sinal do Wiener; maior = mais conservador",
    )
    g.add_argument("--deblur-iters", type=int, default=15, help="iteracoes do Richardson-Lucy")
    g.add_argument(
        "--deblur-min-confidence",
        type=float,
        default=8.0,
        help="confianca minima da estimativa para aplicar a deconvolucao",
    )
    g.add_argument(
        "--deblur-max-length",
        type=int,
        default=40,
        help="comprimento maximo de borrao procurado, em pixels",
    )

    g = p.add_argument_group("detalhe (fora do rosto)")
    g.add_argument("--denoise", type=float, default=0.0, help="0 a 1: reducao de ruido previa")
    g.add_argument(
        "--clahe",
        type=float,
        default=None,
        help="contraste local; 0 desliga (padrao: 1.6 no classic, 0 no modo de IA)",
    )
    g.add_argument("--clahe-grid", type=int, default=8, help="tamanho da grade do CLAHE")
    g.add_argument(
        "--sharpen",
        type=float,
        default=None,
        help="intensidade da nitidez (padrao: 0.7 no classic, 0 no modo de IA)",
    )
    g.add_argument("--sharpen-sigma", type=float, default=1.6, help="raio da nitidez, em px da saida")
    g.add_argument(
        "--sharpen-threshold",
        type=float,
        default=0.012,
        help="limiar que impede realcar ruido (0 a 1)",
    )

    g = p.add_argument_group("rosto")
    g.add_argument(
        "--face-mode",
        default="preserve",
        choices=("preserve", "gentle", "enhance", "off"),
        help=(
            "preserve: rosto so e redimensionado (padrao); "
            "gentle: rosto recebe o deblur, mas nao o realce; "
            "enhance: rosto tratado como o resto; off: nem detecta"
        ),
    )
    g.add_argument("--face-expand", type=float, default=0.30, help="folga da mascara ao redor do rosto")
    g.add_argument("--face-feather", type=float, default=0.30, help="suavidade da borda da mascara")
    g.add_argument("--face-model", help="caminho de um face_detection_yunet_2023mar.onnx")
    g.add_argument(
        "--no-download",
        action="store_true",
        help="nao baixar o modelo de rostos automaticamente",
    )
    g.add_argument(
        "--face-min-size",
        type=float,
        default=0.02,
        help="ignora rostos menores que esta fracao do lado maior",
    )

    g = p.add_argument_group("execucao")
    g.add_argument("--quality", type=int, default=95, help="qualidade JPEG/WebP da saida")
    g.add_argument("--band-height", type=int, default=256, help="linhas por faixa de processamento")
    g.add_argument(
        "--max-pixels",
        type=float,
        default=250.0,
        help="teto de megapixels da saida, como trava de memoria (padrao: 250)",
    )
    g.add_argument("--codec", default="mp4v", help="fourcc do video de saida (padrao: mp4v)")
    g.add_argument("--face-interval", type=int, default=1, help="detectar rostos a cada N quadros")
    g.add_argument("--blur-interval", type=int, default=1, help="reestimar o borrao a cada N quadros")
    g.add_argument("--max-frames", type=int, help="processar no maximo N quadros (teste rapido)")
    g.add_argument("-q", "--quiet", action="store_true", help="nao imprime progresso")
    return p


def config_from_args(args: argparse.Namespace) -> EnhanceConfig:
    # A rede ja devolve a imagem nitida. Repetir CLAHE e unsharp por cima so
    # deixa o resultado duro, entao no modo de IA esses realces saem de fabrica
    # desligados — a menos que o usuario peca.
    com_ia = args.engine in ("ai", "auto")
    clahe = args.clahe if args.clahe is not None else (0.0 if com_ia else 1.6)
    sharpen = args.sharpen if args.sharpen is not None else (0.0 if com_ia else 0.7)

    cfg = EnhanceConfig(
        scale=args.scale,
        interpolation=args.interp,
        engine=args.engine,
        ai_model=args.ai_model,
        ai_tile=args.ai_tile,
        ai_overlap=args.ai_overlap,
        ai_threads=args.ai_threads,
        ai_allow_download=not args.no_download,
        deblur_method=args.deblur,
        deblur_strength=args.deblur_strength,
        deblur_noise=args.deblur_noise,
        deblur_iterations=args.deblur_iters,
        deblur_min_confidence=args.deblur_min_confidence,
        deblur_max_length=args.deblur_max_length,
        denoise=args.denoise,
        clahe_clip=clahe,
        clahe_grid=args.clahe_grid,
        sharpen_sigma=args.sharpen_sigma,
        sharpen_amount=sharpen,
        sharpen_threshold=args.sharpen_threshold,
        face_mode=args.face_mode,
        face_expand=args.face_expand,
        face_feather=args.face_feather,
        face_model=args.face_model,
        face_min_size_ratio=args.face_min_size,
        face_allow_download=not args.no_download,
        band_height=args.band_height,
        max_output_pixels=int(args.max_pixels * 1e6),
        codec=args.codec,
        face_interval=args.face_interval,
        blur_interval=args.blur_interval,
    )
    cfg.validate()
    return cfg


def _default_output(src: str, scale: float, out_dir: Optional[str] = None) -> str:
    stem, ext = os.path.splitext(os.path.basename(src))
    label = f"{scale:g}x".replace(".", "_")
    name = f"{stem}_{label}{ext}"
    return os.path.join(out_dir or os.path.dirname(os.path.abspath(src)), name)


def _collect_inputs(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise IOError(f"entrada nao encontrada: {path}")
    known = IMAGE_EXTS | VIDEO_EXTS
    files = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.splitext(name)[1].lower() in known
    ]
    if not files:
        raise IOError(f"nenhuma imagem ou video em: {path}")
    return files


def _describe(report: Report) -> str:
    w, h = report.input_size
    ow, oh = report.output_size
    parts = [
        f"{w}x{h} -> {ow}x{oh}",
        "IA" if report.engine == "ai" else "classic",
        f"{report.faces} rosto(s)",
    ]
    if report.deblur_applied:
        parts.append(
            f"borrao {report.blur_length:.1f}px @ {report.blur_angle:.0f}deg "
            f"(conf {report.blur_confidence:.1f})"
        )
    else:
        parts.append("sem motion blur detectado")
    parts.append(f"{report.seconds:.1f}s")
    return " | ".join(parts)


def _silenciar_opencv() -> None:
    """Corta os avisos internos do OpenCV, que o usuario nao tem como resolver."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:  # versoes antigas nao expoem o modulo de logging
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _silenciar_opencv()
    try:
        cfg = config_from_args(args)
        inputs = _collect_inputs(args.input)
    except (ValueError, IOError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    batch = len(inputs) > 1 or os.path.isdir(args.input)
    out_dir = args.output if (batch and args.output) else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    detector = None
    if cfg.face_mode != "off":
        try:
            detector = build_detector(cfg)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 2

    resolver = None
    if cfg.engine != "classic":
        try:
            resolver = build_resolver(cfg)
        except RuntimeError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 2
        if resolver is None and not args.quiet:
            print(
                "aviso: modelo de IA indisponivel, usando ampliacao classica "
                "(interpolacao, sem reconstruir detalhe)",
                file=sys.stderr,
            )

    failures = 0
    for src in inputs:
        dst = (
            _default_output(src, cfg.scale, out_dir)
            if batch or not args.output
            else args.output
        )
        if os.path.abspath(dst) == os.path.abspath(src):
            print(f"erro: saida igual a entrada ({src}), pulando", file=sys.stderr)
            failures += 1
            continue

        try:
            if is_video(src):
                if not args.quiet:
                    print(f"[video] {src} -> {dst}")

                def show(i: int, total: int, report: Report, _dst: str = dst) -> None:
                    if args.quiet:
                        return
                    total_txt = str(total) if total else "?"
                    end = "\n" if total and i + 1 == total else "\r"
                    print(f"  quadro {i + 1}/{total_txt} — {_describe(report)}", end=end, flush=True)

                info = process_video_file(
                    src, dst, cfg, progress=show, max_frames=args.max_frames
                )
                if not args.quiet:
                    print(
                        f"\n  {info['frames']} quadros, "
                        f"{info['deblurred_frames']} com deblur aplicado. "
                        "Audio nao e copiado — veja o README."
                    )
            else:
                report = process_image_file(
                    src, dst, cfg, args.quality, detector=detector, resolver=resolver
                )
                if not args.quiet:
                    print(f"[imagem] {src} -> {dst}\n  {_describe(report)}")
        except (IOError, MemoryError, ValueError) as exc:
            print(f"erro em {src}: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
