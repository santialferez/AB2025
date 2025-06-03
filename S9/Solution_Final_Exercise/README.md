# 📊 Ejercicio Final: Grafo de Conocimiento de Prompt Engineering

Este proyecto implementa una solución completa para crear grafos de conocimiento interactivos utilizando las técnicas del blog [Google Gemini LangChain Cheatsheet](https://www.philschmid.de/gemini-langchain-cheatsheet).

## 🎯 Objetivo

Crear un sistema que:
1. Procese audio educativo usando la API de Gemini
2. Extraiga conceptos y relaciones mediante prompt engineering avanzado
3. Estructure la información usando Pydantic
4. Genere visualizaciones interactivas con NetworkX y Plotly

## 🛠️ Tecnologías Utilizadas

- **Google Gemini API** - Procesamiento multimodal (audio, texto)
- **LangChain** - Framework para aplicaciones con LLMs
- **Pydantic** - Validación y estructuración de datos
- **NetworkX** - Análisis y creación de grafos
- **Matplotlib/Plotly** - Visualizaciones estáticas e interactivas
- **Jupyter** - Notebooks para experimentación

## 🚀 Instalación y Uso

### Método 1: Instalación Automática (Recomendado)

```bash
# Ejecutar el script de instalación automática
python setup_and_run.py
```

Este script automáticamente:
- ✅ Instala todas las dependencias necesarias
- ✅ Configura tu API key de Google AI
- ✅ Ejecuta una demostración con datos de ejemplo
- ✅ Te guía paso a paso

### Método 2: Instalación Manual

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API Key
export GOOGLE_API_KEY="tu_api_key_aquí"

# 3. Ejecutar demo rápido
python simple_demo.py

# 4. Procesamiento completo del audio (opcional)
python knowledge_graph_creator.py
```

### 3. Obtener API Key

Obtén tu API key gratis desde [Google AI Studio](https://aistudio.google.com/):
- Es completamente gratuito
- No requiere tarjeta de crédito
- Acceso inmediato

## 📁 Estructura del Proyecto

```
ABprompt/
├── knowledge_graph_creator.py      # ⚡ Implementación principal
├── simple_demo.py                  # 🚀 Demo rápido con texto de ejemplo
├── setup_and_run.py               # 🔧 Instalación automática
├── requirements.txt               # 📦 Dependencias
├── README.md                     # 📖 Documentación principal
├── IMPLEMENTATION_SUMMARY.md     # 📋 Resumen técnico detallado
├── 2025-06-02_15-06-35_audio.mp3 # 🎵 Audio de la clase original
├── transcription.txt             # 📄 Transcripción generada
├── knowledge_graph_matplotlib.png # 📊 Visualización estática
└── knowledge_graph_interactive.html # 🌐 Visualización interactiva
```

## 🧠 Técnicas de Prompt Engineering Implementadas

### 1. **Chain of Thought (CoT)**
```python
INSTRUCCIONES PASO A PASO (Chain of Thought):
1. LEE completamente la transcripción
2. IDENTIFICA todos los conceptos de prompt engineering mencionados
3. CLASIFICA cada concepto por categoría
4. EVALÚA la importancia de cada concepto (1-5)
5. DETECTA relaciones entre conceptos
6. ESTRUCTURA toda la información en formato JSON
```

### 2. **Few-Shot Learning**
Proporciona ejemplos estructurados de conceptos antes del análisis:
```python
EJEMPLOS = {
    "nombre": "Few-shot Learning",
    "definicion": "Técnica que proporciona ejemplos al modelo",
    "categoria": "tecnica",
    "importancia": 4,
    "ejemplos": ["Dar 3 ejemplos antes de la tarea real"]
}
```

### 3. **Structured Output con Pydantic**
```python
class Concepto(BaseModel):
    nombre: str = Field(..., description="The concept name")
    definicion: str = Field(..., description="Definition of the concept")
    categoria: str = Field(..., description="Category type")
    importancia: int = Field(..., description="Importance level 1-5")
    ejemplos: Optional[List[str]] = Field(default=[])
```

### 4. **Zero-Shot con Instrucciones Claras**
Definiciones precisas de categorías y tipos de relación para análisis sin ejemplos previos.

## 📊 Modelo de Datos

### Conceptos
- **nombre**: Nombre del concepto
- **definicion**: Descripción detallada
- **categoria**: `"tecnica"`, `"herramienta"`, `"metodologia"`, `"concepto"`
- **importancia**: 1-5 (1=básico, 5=avanzado)
- **ejemplos**: Lista de ejemplos de uso

### Relaciones
- **concepto_origen/destino**: Nombres de los conceptos conectados
- **tipo_relacion**: `"es_parte_de"`, `"requiere"`, `"mejora"`, `"se_aplica_con"`
- **fuerza**: 0.0-1.0 (fuerza de la relación)

## 🎨 Visualizaciones Generadas

### 1. **Grafo Estático (Matplotlib)**
- Nodos coloreados por categoría
- Tamaño proporcional a importancia
- Aristas con grosor según fuerza de relación
- Layout optimizado con algoritmo spring

### 2. **Grafo Interactivo (Plotly)**
- Hover para ver detalles de conceptos
- Zoom y pan interactivo
- Leyenda por categorías
- Exportable como HTML standalone

## 📈 Modos de Uso

### 🚀 Demo Rápido (Recomendado para empezar)
```python
# Ejecutar demo con texto de ejemplo
python simple_demo.py
```
- ⚡ Rápido (menos de 1 minuto)
- 🎯 Ideal para probar el sistema
- 📝 Usa contenido educativo predefinido
- ✅ No requiere archivos de audio grandes

### 🎵 Procesamiento de Audio Completo
```python
# Procesar el audio original de la clase
python knowledge_graph_creator.py
```
- 🕐 Toma varios minutos (transcripción + análisis)
- 🎯 Análisis completo del contenido real
- 🎵 Procesa audio de 119MB
- 📊 Genera análisis más profundo

### 🔧 Instalación Guiada
```python
# Instalación automática con guía paso a paso
python setup_and_run.py
```

## 📁 Archivos Generados

Después de ejecutar el sistema, se generan:

- `knowledge_graph_data.json` - Datos estructurados del grafo
- `knowledge_graph_report.md` - Reporte detallado en Markdown
- `knowledge_graph_matplotlib.png` - Visualización estática
- `knowledge_graph_interactive.html` - Visualización interactiva
- `transcription.txt` - Transcripción del audio

## 🔧 Funcionalidades Avanzadas

### Procesamiento Multimodal
```python
# Audio input (implementado según el blog post)
message = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "media", "data": encoded_audio, "mime_type": "audio/mpeg"}
    ]
)
```

### Structured Output
```python
# Using structured LLM (implementado según el blog post)
structured_llm = llm.with_structured_output(GrafoConocimiento)
result = structured_llm.invoke({"transcripcion": text})
```

### Optimización de Rendimiento
```python
# Reutilización inteligente de transcripciones
if os.path.exists('transcription.txt'):
    print("📄 Found existing transcription, loading...")
    # Evita re-transcribir el mismo audio
```

## 🧪 Testing y Validación

### Validar Estructura de Datos
```python
# Pydantic automáticamente valida la estructura
try:
    grafo = GrafoConocimiento(**data)
    print("✅ Estructura válida")
except ValidationError as e:
    print(f"❌ Error: {e}")
```

### Verificar Calidad del Grafo
```python
# Estadísticas automáticas del grafo
print(f"Nodos: {G.number_of_nodes()}")
print(f"Aristas: {G.number_of_edges()}")
print(f"Densidad: {nx.density(G):.3f}")
```

## 🚨 Solución de Problemas

### Error de API Key
```
❌ Error: API key not found
```
**Solución**: 
1. Ejecuta `python setup_and_run.py` para configuración guiada
2. O configura manualmente: `export GOOGLE_API_KEY="tu_key"`
3. Obtén tu key gratis en [Google AI Studio](https://aistudio.google.com/)

### Error de Dependencias
```
❌ ModuleNotFoundError: No module named 'langchain_google_genai'
```
**Solución**: 
1. Ejecuta `python setup_and_run.py` para instalación automática
2. O instala manualmente: `pip install -r requirements.txt`

### Error de Procesamiento de Audio
```
❌ Error: File too large
```
**Solución**: 
1. Usa `simple_demo.py` para pruebas rápidas
2. Verifica que el archivo sea < 100MB
3. Usa formatos compatibles: MP3, WAV, M4A

### Problemas de Visualización
```
❌ Error: Cannot display interactive plot
```
**Solución**: 
1. Los archivos HTML se generan en el directorio actual
2. Abre `knowledge_graph_interactive.html` en tu navegador
3. Para Jupyter: asegúrate de tener `ipywidgets` instalado

## 📚 Referencias y Recursos

### Documentación Técnica
- [Google Gemini LangChain Cheatsheet](https://www.philschmid.de/gemini-langchain-cheatsheet) - Fuente principal
- [LangChain Documentation](https://python.langchain.com/) - Framework oficial
- [Google AI Studio](https://aistudio.google.com/) - Para obtener API keys
- [Pydantic Documentation](https://docs.pydantic.dev/) - Validación de datos
- [NetworkX Documentation](https://networkx.org/) - Análisis de grafos
- [Plotly Documentation](https://plotly.com/python/) - Visualizaciones interactivas

### Técnicas Implementadas
- **Chain of Thought**: Razonamiento paso a paso
- **Few-Shot Learning**: Aprendizaje con pocos ejemplos
- **Zero-Shot Learning**: Instrucciones sin ejemplos
- **Structured Output**: Salidas estructuradas con Pydantic
- **Multimodal Processing**: Procesamiento de audio y texto

## 🎉 Competencias Demostradas

✅ **Prompt Engineering Avanzado**
- Chain of Thought para razonamiento estructurado
- Few-shot learning con ejemplos contextuales
- Zero-shot learning con instrucciones precisas
- Structured output con validación automática

✅ **Integración de APIs Modernas**
- Google Gemini API con LangChain
- Procesamiento multimodal (audio → texto)
- Manejo robusto de errores y reintentos

✅ **Estructuración de Datos**
- Modelos Pydantic complejos y validados
- JSON estructurado con tipos de datos
- Relaciones jerárquicas entre conceptos

✅ **Visualización de Datos**
- Grafos estáticos con Matplotlib
- Visualizaciones interactivas con Plotly
- Análisis de redes con NetworkX

✅ **Experiencia de Usuario**
- Scripts de instalación automática
- Múltiples modos de uso (demo/producción)
- Documentación completa y clara
- Manejo de errores informativo

## 🏆 Características Destacadas

- 🎯 **Plug & Play**: Ejecuta `python setup_and_run.py` y funciona
- ⚡ **Demo Instantáneo**: Prueba sin audio con `simple_demo.py`
- 🎵 **Procesamiento Real**: Analiza audio educativo completo
- 📊 **Visualizaciones Duales**: Estática + Interactiva
- 🔄 **Reutilización Inteligente**: Evita re-procesar archivos existentes
- 📚 **Documentación Completa**: README + Implementation Summary
- 🛡️ **Validación Automática**: Pydantic garantiza estructura correcta
- 🌐 **Exportación Web**: HTML standalone para compartir resultados

---

**💡 ¿Primera vez usándolo?** Ejecuta `python setup_and_run.py` para una experiencia guiada completa.

**⚡ ¿Solo quieres ver cómo funciona?** Ejecuta `python simple_demo.py` para una demostración rápida.

**🎵 ¿Quieres análisis completo?** Ejecuta `python knowledge_graph_creator.py` con tu API key configurada.