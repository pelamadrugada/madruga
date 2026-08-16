# nitido

Amplia imagens e vídeos em **5×**, tira o **motion blur** e aumenta o detalhe
de tudo ao redor — **sem alterar o rosto**.

O rosto é detectado, protegido por uma máscara suave e apenas
*redimensionado*: nenhuma deconvolução, nenhum contraste local e nenhuma
nitidez artificial encostam nele. Todo o resto do quadro passa pelo
tratamento completo. A emenda entre as duas partes é gradual, então não
aparece costura.

```
FUNDO (pena do chapéu)          ROSTO (olho)
Lanczos 5x  →  nitido 5x        Lanczos 5x  →  nitido 5x
  borrado      com textura        idêntico     idêntico
```

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Só precisa de `opencv-python-headless` e `numpy`.

Na primeira execução o programa baixa uma vez (≈230 KB) o modelo de detecção
de rostos **YuNet** para `~/.cache/nitido/`. Sem rede, baixe
[`face_detection_yunet_2023mar.onnx`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
e use `--face-model CAMINHO`, ou aponte `NITIDO_FACE_MODEL` para ele.

## Uso

```bash
# uma foto — grava foto_5x.jpg ao lado do original
.venv/bin/python -m nitido foto.jpg

# escolhendo a saída
.venv/bin/python -m nitido foto.jpg -o grande.png

# uma pasta inteira
.venv/bin/python -m nitido entrada/ -o saida/

# vídeo (sem áudio — veja a observação abaixo)
.venv/bin/python -m nitido clipe.mp4 -o clipe_5x.mp4
```

Saída típica:

```
[imagem] tremida.png -> tremida_5x.png
  512x512 -> 2560x2560 | 1 rosto(s) | borrao 11.7px @ 20deg (conf 14.7) | 2.6s
```

## Como o rosto é preservado

1. **Detecção** — YuNet acha os rostos (`nitido/faces.py`).
2. **Máscara** — cada rosto vira uma elipse um pouco maior que a caixa, para
   pegar cabelo, orelhas e queixo, com borda suavizada (`face_mask`).
3. **Dois caminhos** — o fundo recebe deblur + contraste local + nitidez; o
   rosto recebe só o redimensionamento.
4. **Composição** — os dois caminhos se misturam pela máscara, ainda em
   resolução original, e só então tudo sobe para 5×.

Na prática o núcleo do rosto sai com diferença máxima de **1/255** em relação
a um Lanczos puro do original — é o mesmo pixel, só maior. Isso é verificado
em teste (`test_rosto_fica_igual_ao_simples_redimensionamento`).

Se preferir outro comportamento:

| `--face-mode` | o que acontece com o rosto                         |
|---------------|----------------------------------------------------|
| `preserve`    | só redimensionado (**padrão**)                     |
| `gentle`      | recebe a remoção de motion blur, mas não o realce  |
| `enhance`     | tratado como o resto do quadro                     |
| `off`         | nem detecta rosto (mais rápido, sem proteção)      |

## Como o motion blur é removido

O borrão de movimento é uma convolução por um segmento de reta. O programa
estima o **comprimento** e o **ângulo** desse segmento pelo cepstro da imagem
e desfaz a convolução na luminância (croma fica de fora para não criar franja
colorida).

A estimativa só é aplicada quando a confiança passa de `8` desvios-padrão.
Em medições com imagens sintéticas, motion blur real dá confiança de 11 a 27,
enquanto foto nítida ou fora de foco fica em torno de 3 — a margem é
folgada de propósito, porque deconvoluir uma foto que já estava boa só
adiciona anel.

| flag                | efeito                                                     |
|---------------------|------------------------------------------------------------|
| `--deblur wiener`   | rápido, padrão                                             |
| `--deblur rl`       | Richardson-Lucy: mais lento, menos anel                     |
| `--deblur none`     | desliga                                                    |
| `--deblur-strength` | 0 a 1, quanto do resultado entra (padrão 0.85)             |
| `--deblur-noise`    | maior = mais conservador (menos anel, menos nitidez)       |

Fora de foco não é motion blur: não há direção para desfazer, então o
programa deliberadamente não mexe nesses casos.

## Detalhe do que não é rosto

| flag                  | efeito                                            |
|-----------------------|---------------------------------------------------|
| `--clahe 1.6`         | contraste local; `0` desliga                      |
| `--sharpen 0.7`       | intensidade da máscara de nitidez                 |
| `--sharpen-sigma 1.6` | raio da nitidez, em pixels da **saída**           |
| `--sharpen-threshold` | limiar que impede transformar ruído em granulado  |
| `--denoise 0.0`       | redução de ruído prévia, de 0 a 1                 |

## Memória e tamanho

5× multiplica a área por **25**: uma foto de 12 MP vira 300 MP. Por isso a
etapa em alta resolução roda em faixas horizontais com sobreposição, e existe
uma trava padrão de 250 MP na saída (`--max-pixels`). O resultado das faixas
é idêntico ao da imagem inteira — também coberto por teste.

## Vídeo

Funciona quadro a quadro, com detecção de rosto por quadro. Duas observações:

* **O áudio não é copiado** (o OpenCV não lê trilha de áudio). Para juntar de
  volta:
  ```bash
  ffmpeg -i clipe_5x.mp4 -i clipe.mp4 -c copy -map 0:v:0 -map 1:a:0 final.mp4
  ```
* 5× em vídeo gera arquivos enormes e demora. Para ir mais rápido:
  `--face-interval 5` (detecta rosto a cada 5 quadros), `--blur-interval 10`
  (reaproveita a estimativa de borrão) e `--max-frames N` para testar num
  trecho antes de rodar tudo.

## Testes

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

Os testes não usam rede. Dois deles pulam por padrão e só rodam com um modelo
e uma foto de referência à mão:

```bash
NITIDO_TEST_FACE_MODEL=~/.cache/nitido/face_detection_yunet_2023mar.onnx \
NITIDO_TEST_FACE_IMAGE=retrato.jpg \
.venv/bin/python -m pytest -q
```

## Estrutura

```
nitido/faces.py     detecção de rostos e máscara de proteção
nitido/models.py    resolução/download do modelo YuNet
nitido/deblur.py    estimativa do borrão e deconvolução
nitido/enhance.py   contraste local, unsharp, redimensionamento
nitido/pipeline.py  orquestração (imagem e vídeo)
nitido/cli.py       linha de comando
```

## Atalho

Se não quiser mexer em venv, use o script — ele prepara tudo na primeira vez:

```bash
./rodar.sh foto.jpg                  # grava foto_5x.jpg ao lado
./rodar.sh foto.jpg -o grande.png
./rodar.sh pasta/ -o saida/
./rodar.sh clipe.mp4 -o clipe_5x.mp4
```
