# Guía Paso a Paso: Gemini Embedding 2 — Embeddings Multimodales

> **Para:** Maestría en Analítica de Datos — Curso de NLP  
> **Última actualización:** Abril 2026 (modelo en GA desde abril 2026)

---

## ¿Qué es Gemini Embedding 2?

`gemini-embedding-2` es el primer modelo de embeddings **nativo multimodal** de Google, construido sobre la arquitectura Gemini. Convierte texto, imágenes, video, audio y documentos PDF en vectores numéricos dentro de un **único espacio semántico compartido**, lo que permite comparar y buscar entre diferentes tipos de contenido.

---

## Paso 1 — Obtener una API Key

1. Ve a **[Google AI Studio](https://aistudio.google.com/apikey)**
2. Inicia sesión con tu cuenta de Google
3. Haz clic en **"Create API Key"**
4. Selecciona o crea un proyecto de Google Cloud
5. Copia la API key generada — guárdala en un lugar seguro

> ⚠️ **Nunca** incluyas la API key directamente en tu código si lo vas a compartir o subir a GitHub.

---

## Paso 2 — Instalar el SDK

```bash
# Instalar el SDK oficial de Google GenAI
pip install google-genai

# Para el notebook completo, también necesitas:
pip install Pillow matplotlib scikit-learn requests numpy
```

### Verificar instalación

```python
from google import genai
print("✅ SDK instalado correctamente")
```

---

## Paso 3 — Configurar la API Key

### Opción A: Variable de entorno (recomendado)

```bash
# En terminal (Linux/Mac)
export GEMINI_API_KEY="tu_api_key_aqui"

# En Windows (PowerShell)
$env:GEMINI_API_KEY = "tu_api_key_aqui"
```

### Opción B: Google Colab Secrets (recomendado en Colab)

1. En Colab, abre el panel lateral izquierdo → ícono de llave 🔑
2. Agrega un secreto con nombre `GEMINI_API_KEY` y tu key como valor
3. En el notebook:

```python
from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
```

### Opción C: Archivo `.env` (para proyectos locales)

```bash
# Crear archivo .env en la raíz del proyecto
echo "GEMINI_API_KEY=tu_api_key_aqui" > .env
```

```python
# En Python
from dotenv import load_dotenv
load_dotenv()
```

---

## Paso 4 — Primera llamada a la API

```python
from google import genai

# El cliente lee automáticamente la variable de entorno GEMINI_API_KEY
client = genai.Client()

# Generar embedding de texto
resultado = client.models.embed_content(
    model="gemini-embedding-2",
    contents="El aprendizaje automático es fascinante"
)

vector = resultado.embeddings[0].values
print(f"Dimensiones del vector: {len(vector)}")  # → 3072
print(f"Primeras 5 dims: {vector[:5]}")
```

---

## Paso 5 — Usar Task Prefixes (búsqueda asimétrica)

Para búsqueda de documentos, el rendimiento mejora significativamente cuando se indica la tarea:

```python
# Para consultas (queries del usuario)
def preparar_query(texto):
    return f"task: search result | query: {texto}"

# Para documentos del corpus (lo que se indexa)
def preparar_documento(titulo, texto):
    return f"title: {titulo} | text: {texto}"

# Generar embeddings
emb_query = client.models.embed_content(
    model="gemini-embedding-2",
    contents=preparar_query("¿Cómo funciona una red neuronal?")
)

emb_doc = client.models.embed_content(
    model="gemini-embedding-2",
    contents=preparar_documento(
        "Redes Neuronales Artificiales",
        "Una red neuronal está compuesta por capas de nodos interconectados..."
    )
)
```

### Tabla de prefijos disponibles

| Tarea | Prefijo de query | Estructura del documento |
|---|---|---|
| Búsqueda web | `task: search result \| query: {q}` | `title: {t} \| text: {c}` |
| Preguntas y respuestas | `task: question answering \| query: {q}` | `title: {t} \| text: {c}` |
| Verificación de hechos | `task: fact checking \| query: {q}` | `title: {t} \| text: {c}` |
| Recuperación de código | `task: code retrieval \| query: {q}` | `title: {t} \| text: {c}` |

---

## Paso 6 — Embeddings de Imágenes

Las imágenes se envían como bytes con su tipo MIME:

```python
from google.genai import types

# Cargar imagen como bytes
with open("mi_imagen.jpg", "rb") as f:
    imagen_bytes = f.read()

# Generar embedding de la imagen
resultado = client.models.embed_content(
    model="gemini-embedding-2",
    contents=types.Part.from_bytes(
        data=imagen_bytes,
        mime_type="image/jpeg"  # o "image/png"
    )
)

vector_imagen = resultado.embeddings[0].values
print(f"Vector de imagen: {len(vector_imagen)} dims")
```

### Formatos soportados

| Tipo | Formatos | Límite |
|---|---|---|
| Imágenes | JPEG, PNG | Hasta 6 imágenes por request |
| Video | MP4, MOV | Hasta 120 segundos |
| Audio | MP3, WAV, OGG, etc. | Hasta 180 segundos |
| Documentos | PDF | Hasta 6 páginas |
| Tokens de texto | — | Hasta 8192 tokens |

---

## Paso 7 — Embeddings Multimodales (texto + imagen)

Combinar múltiples modalidades en un solo vector:

```python
# Un solo vector que combina texto e imagen
resultado = client.models.embed_content(
    model="gemini-embedding-2",
    contents=[
        "Descripción de la imagen: un perro dorado en un parque",
        types.Part.from_bytes(data=imagen_bytes, mime_type="image/jpeg")
    ]
)

# El resultado es UN SOLO vector agregado
vector_multimodal = resultado.embeddings[0].values
```

> **Nota importante:** Gemini Embedding 2 produce un **único vector agregado** cuando se pasan múltiples inputs. Si necesitas vectores separados para cada item, usa la Batch API.

---

## Paso 8 — Calcular Similitud

```python
import numpy as np

def similitud_coseno(v1, v2):
    """Calcula la similitud coseno entre dos vectores."""
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Comparar texto con imagen (cross-modal!)
emb_texto = client.models.embed_content(
    model="gemini-embedding-2",
    contents="un perro dorado jugando en el parque"
).embeddings[0].values

emb_img = client.models.embed_content(
    model="gemini-embedding-2",
    contents=types.Part.from_bytes(data=imagen_bytes, mime_type="image/jpeg")
).embeddings[0].values

sim = similitud_coseno(emb_texto, emb_img)
print(f"Similitud texto ↔ imagen: {sim:.4f}")
# Un valor cercano a 1.0 indica alta similitud semántica
```

> **Buenas prácticas con similitud:**
> - Usa **ranking** (ordenar por score) en lugar de umbrales fijos
> - El dot product también es válido si los vectores están normalizados
> - Evita usar un valor fijo como "0.7 = relevante" — varía por dominio

---

## Paso 9 — Reducir Dimensionalidad con MRL

Gemini Embedding 2 usa Matryoshka Representation Learning: puedes truncar el vector sin reentrenar:

```python
from google.genai import types

# Dimensiones recomendadas: 128, 256, 512, 768, 1536, 3072 (defecto)
resultado = client.models.embed_content(
    model="gemini-embedding-2",
    contents="Mi texto aquí",
    config=types.EmbedContentConfig(output_dimensionality=768)
)

vector_compacto = resultado.embeddings[0].values
print(f"Dimensiones: {len(vector_compacto)}")  # → 768
```

### Guía de selección de dimensionalidad

| Dimensiones | Caso de uso | Balance |
|---|---|---|
| 128–256 | Prototipado rápido, datasets pequeños | Mínimo costo de almacenamiento |
| 512–768 | Producción con restricciones de memoria | Buen balance calidad/costo |
| 1536 | Producción general | Alta calidad, costo moderado |
| 3072 | Máxima precisión | Por defecto, mayor costo |

---

## Paso 10 — Uso con Batch API (para grandes volúmenes)

Para indexar miles de documentos, la Batch API ofrece 50% de descuento y mayor throughput:

```python
# La Batch API se usa a través de la API estándar con configuración de lote
# Consulta la documentación oficial para el uso completo:
# https://ai.google.dev/gemini-api/docs/batch-api#batch-embedding
```

---

## Arquitectura Recomendada para un Sistema RAG Multimodal

```
┌─────────────────────────────────────────────────────────┐
│                    FASE DE INDEXACIÓN                    │
│                                                          │
│  Documento/Imagen → Gemini Embedding 2 → Vector 3072D    │
│                          ↓                              │
│                    Base de Datos Vectorial               │
│              (ChromaDB / Pinecone / Qdrant)              │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                    FASE DE CONSULTA                      │
│                                                          │
│  Query (texto/imagen) → Gemini Embedding 2 → Vector      │
│                          ↓                              │
│              Búsqueda por similitud coseno               │
│                          ↓                              │
│              Top-K documentos recuperados                │
│                          ↓                              │
│          [Opcional] Gemini para generar respuesta        │
└─────────────────────────────────────────────────────────┘
```

---

## Errores Comunes y Soluciones

| Error | Causa | Solución |
|---|---|---|
| `AuthenticationError` | API key inválida o no configurada | Verificar variable de entorno `GEMINI_API_KEY` |
| `ResourceExhausted` | Límite de tasa excedido | Agregar delays entre llamadas o usar Batch API |
| Similitudes muy bajas en búsqueda | No se usan task prefixes | Agregar `task: search result \| query:` a las consultas |
| Embeddings incompatibles | Mezcla de modelos distintos | No mezclar `gemini-embedding-001` con `gemini-embedding-2` |
| Video/audio no procesado | SDK desactualizado | Actualizar con `pip install --upgrade google-genai` |

---

## Costos y Límites (Gemini API — abril 2026)

Consulta siempre los precios actualizados en [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) ya que pueden cambiar.

- **Batch API** ofrece 50% de descuento en embeddings
- El tier gratuito de Google AI Studio tiene límites de tasa diarios
- Para producción, se recomienda habilitar facturación en Google Cloud

---

## Recursos Adicionales

- 📖 [Documentación oficial de Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- 🔑 [Obtener API Key en Google AI Studio](https://aistudio.google.com/apikey)
- 📓 [Cookbook de Gemini (notebooks oficiales)](https://github.com/google-gemini/cookbook)
- 📊 [Vertex AI — Gemini Embedding 2](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings)
- 📝 [Blog de lanzamiento — Gemini Embedding 2](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- 🛠️ [Referencia del modelo](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2-preview)
