# Traductor de voz

Mandas un audio (una nota de voz, una grabación) y te lo devuelve **dicho por ti en francés,
alemán o inglés**: con tu voz, tu tono, tu ritmo y la emoción de cada frase.

Todo corre en tu propia computadora con modelos abiertos. Después de la primera descarga de los
modelos no necesita internet ni manda tu voz a ningún servicio.

## Cómo funciona

```
tu audio ──► Whisper ──► frases con tiempos ──► traducción ──► voz clonada ──► ajuste de prosodia ──► audio final
                 │                                                ▲                  ▲
                 └──── medición de prosodia de cada frase ─────────┴──────────────────┘
                      (tono, variación de entonación, volumen, duración)
```

1. **Reconocimiento** (faster-whisper): transcribe y corta el audio en frases con sus tiempos.
2. **Medición de prosodia**: por cada frase mide tu tono medio, cuánto varía tu entonación,
   el volumen y la duración. Con eso calcula un **nivel de emoción** (calmado, neutro, animado,
   muy expresivo).
3. **Traducción** (MarianMT, o NLLB-200 para otros pares): frase por frase. Puedes corregir
   cualquier traducción en la app y regenerar.
4. **Voz clonada** (Chatterbox Multilingual): cada frase traducida se genera usando como
   referencia **el audio original de esa misma frase**, así el modelo copia la emoción de ese
   momento (si gritaste, grita; si susurraste, susurra). La intensidad emocional del modelo se
   ajusta con la emoción medida en el paso 2.
   Si guardas un **perfil de voz**, el timbre sale de esa grabación limpia y la emoción sigue
   saliendo de cada frase.
5. **Ajuste de prosodia**: a cada frase generada se le corrige el tono medio (para que tenga tu
   altura de voz en ese momento), el ritmo (para que dure parecido a la original) y el volumen.
   Después se vuelven a poner tus pausas entre frases.

### Qué esperar (con honestidad)

- **Timbre y color de voz**: muy parecidos con 10-20 s de buena referencia. Mejora mucho con un
  perfil grabado sin ruido.
- **Emoción y energía**: se transfieren frase por frase. Las emociones fuertes (alegría,
  enojo, tristeza marcada) se notan bien; los matices finos dependen del modelo.
- **Entonación**: se copia el tono medio y la "energía" de la entonación de cada frase, no la
  curva exacta sílaba por sílaba (los idiomas tienen otras sílabas y otra melodía).
- **Acento**: el control *Acento* va de 0 (acento nativo francés/alemán/inglés) a 0.6 (más tu
  propio acento). El punto medio suele sonar más natural.

## Instalación en tu computadora

Necesitas **Python 3.11**, **ffmpeg** y, para que vaya rápido, una **GPU NVIDIA con 8 GB de
VRAM o más**. Sin GPU funciona igual, pero tarda varias veces la duración del audio por idioma.

### Windows

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg          # cierra y vuelve a abrir la terminal después

cd traductor-voz
py -3.11 -m venv .venv
.venv\Scripts\activate

# PyTorch con soporte de GPU NVIDIA (Chatterbox necesita la versión 2.6.0)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[chatterbox]"
traductor-voz app
```

Se abre en <http://127.0.0.1:7860>. La primera vez descarga los modelos (varios GB), tarda.

> Las placas RTX serie 50 necesitan PyTorch 2.7 o más nuevo, que Chatterbox todavía no acepta
> oficialmente. Si tienes una, avísame y armamos la instalación a mano.

### Linux / macOS

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[chatterbox]"
traductor-voz app
```

En Mac con Apple Silicon usa la GPU (MPS) automáticamente.

## Uso

### App web

1. Pestaña **Mi perfil de voz** (opcional, recomendado): lee en voz alta 10-20 s con tu tono
   normal, en un lugar sin ruido, y guárdalo.
2. Pestaña **Traducir**: graba o sube tu audio (sirven las notas de voz de WhatsApp, `.ogg`),
   elige los idiomas, elige tu perfil y toca **Traducir**.
3. En **Textos** puedes corregir cualquier traducción y tocar **Regenerar voz con estos textos**.
4. En **Emoción detectada en cada frase** ves cómo interpretó cada frase y qué ajustes aplicó.

Ajustes:

| Ajuste | Qué hace |
|---|---|
| Intensidad emocional | Cuánta emoción pone el modelo (0.5 = normal; súbelo si suena plano). |
| Acento | 0 = acento nativo del idioma destino; 0.6 = conserva tu acento. |
| Copiar tu tono | Corrige la altura de la voz de cada frase para que sea la tuya en ese momento. |
| Copiar tu ritmo | Acerca la duración de cada frase a la original (0 = ritmo libre). |
| Respetar mis pausas | Mantiene tus silencios entre frases. |

Para usarla desde el celular en tu casa: `traductor-voz app --host 0.0.0.0` y entra a
`http://IP-de-tu-PC:7860`. El navegador sólo deja grabar con el micrófono en `localhost` o con
https; desde el celular sube el audio como archivo, o usa `--compartir` para un enlace https
temporal.

### Línea de comandos

```bash
# Traduce a francés, alemán e inglés (por defecto) y guarda en ./salida
traductor-voz traducir nota.ogg

# Sólo inglés, en mp3, con tu perfil y más emoción
traductor-voz traducir nota.ogg -i en -f mp3 --perfil yo --intensidad 0.7

# Perfiles
traductor-voz perfil crear yo lectura.wav
traductor-voz perfil listar
```

Los perfiles se guardan en `~/.traductor-voz/perfiles` (cámbialo con la variable
`TRADUCTOR_VOZ_DATOS`).

## Motores de voz

| Motor | Licencia | Idiomas | Emoción | Notas |
|---|---|---|---|---|
| **Chatterbox Multilingual** (por defecto) | MIT | 23 (incluye fr, de, en, es) | Control explícito de intensidad | Agrega una marca de agua inaudible que identifica el audio como generado por IA. |
| **Qwen3-TTS Base** | Apache 2.0 | 10 (incluye fr, de, en, es) | Sólo la que copia de la referencia | Clona con 3 s de voz. Necesita su propio entorno. |

Cada motor fija versiones distintas de `transformers`, así que van en entornos separados:

```bash
python3.11 -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install -e ".[qwen]"
traductor-voz app --motor qwen3
```

Vale la pena probar los dos con tu voz y quedarte con el que suene más a ti en cada idioma.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

Las pruebas usan motores simulados y voces sintéticas: no descargan modelos.

## Correr en la nube (Claude Code)

En el entorno en la nube se puede desarrollar y correr las pruebas. Para probar con los modelos
reales hace falta permitir `huggingface.co` (y sus dominios de descarga, como
`cdn-lfs.huggingface.co` y `*.hf.co`) en el acceso de red del entorno. Ese entorno no tiene GPU,
así que sirve para audios cortos de prueba, no para el uso diario.

## Próximos pasos posibles

- Bot de Telegram o WhatsApp: le mandas la nota de voz y te responde con las traducciones.
- Traducción con un modelo de lenguaje local (más natural con modismos y frases coloquiales).
- Ajuste fino (*fine-tuning*) del modelo de voz con 30-60 min de tu voz para un clon todavía
  más fiel.
- Limpieza de ruido del audio de entrada.

## Uso responsable

Clona sólo tu propia voz, o la de alguien que te haya dado permiso. Las voces generadas con
Chatterbox llevan una marca de agua inaudible (Perth) que permite detectar que son de IA.
