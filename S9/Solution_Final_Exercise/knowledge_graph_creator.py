import getpass
import os
import base64
from typing import List, Optional
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json

# Configure Plotly to use browser renderer for standalone scripts
pio.renderers.default = "browser"

# Set up Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Pydantic models for structured output
class Concepto(BaseModel):
    """Information about a prompt engineering concept."""
    nombre: str = Field(..., description="The concept name")
    definicion: str = Field(..., description="Definition of the concept")
    categoria: str = Field(..., description="Category: 'tecnica', 'herramienta', 'metodologia', 'concepto'")
    importancia: int = Field(..., description="Importance level 1-5 (1=basic, 5=advanced)")
    ejemplos: Optional[List[str]] = Field(default=[], description="Examples of the concept")

class Relacion(BaseModel):
    """Relationship between two concepts."""
    concepto_origen: str = Field(..., description="Source concept name")
    concepto_destino: str = Field(..., description="Target concept name")
    tipo_relacion: str = Field(..., description="Relationship type: 'es_parte_de', 'requiere', 'mejora', 'se_aplica_con'")
    fuerza: float = Field(..., description="Relationship strength 0.0-1.0")

class GrafoConocimiento(BaseModel):
    """Complete knowledge graph structure."""
    conceptos: List[Concepto]
    relaciones: List[Relacion]
    tema_principal: str = Field(..., description="Main topic of the knowledge graph")
    resumen: str = Field(..., description="Summary of the content")

class KnowledgeGraphCreator:
    def __init__(self):
        # Initialize Gemini model following the blog post pattern
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-05-06",
            temperature=0.7,
            max_retries=2,
        )
        
        # Create structured LLM for JSON output
        self.structured_llm = self.llm.with_structured_output(GrafoConocimiento)
        
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using Gemini API following the blog post example."""
        
        # Check if transcription already exists
        if os.path.exists('transcription.txt'):
            print("📄 Found existing transcription file, loading from 'transcription.txt'...")
            with open('transcription.txt', 'r', encoding='utf-8') as f:
                transcription = f.read()
            print("✅ Transcription loaded from existing file!")
            return transcription
        
        print("🎵 Transcribing audio file...")
        
        audio_mime_type = "audio/mpeg"
        
        with open(audio_file_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe this audio file completely and accurately. Focus on educational content about prompt engineering, AI, and related technical concepts."},
                {"type": "media", "data": encoded_audio, "mime_type": audio_mime_type}
            ]
        )
        
        response = self.llm.invoke([message])
        transcription = response.content
        
        # Save transcription to file
        with open('transcription.txt', 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        print("✅ Audio transcription completed and saved to 'transcription.txt'!")
        return transcription
    
    def create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create comprehensive analysis prompt using Chain of Thought technique."""
        return ChatPromptTemplate.from_messages([
            ("system", """Eres un experto en análisis de contenido educativo especializado en Prompt Engineering. 
            Tu tarea es crear un grafo de conocimiento detallado y estructurado.
            
            INSTRUCCIONES PASO A PASO (Chain of Thought):
            1. LEE completamente la transcripción
            2. IDENTIFICA todos los conceptos de prompt engineering mencionados
            3. CLASIFICA cada concepto por categoría (técnica, herramienta, metodología, concepto)
            4. EVALÚA la importancia de cada concepto (1-5)
            5. DETECTA relaciones entre conceptos
            6. ESTRUCTURA toda la información en el formato JSON requerido
            
            CATEGORÍAS VÁLIDAS:
            - "tecnica": Técnicas específicas (Few-shot, Zero-shot, Chain of Thought, etc.)
            - "herramienta": Herramientas y APIs (LangChain, Gemini, Pydantic, etc.)
            - "metodologia": Metodologías y enfoques (Prompt Engineering, RAG, etc.)
            - "concepto": Conceptos teóricos (LLM, Token, Embedding, etc.)
            
            TIPOS DE RELACIÓN:
            - "es_parte_de": Un concepto es componente de otro
            - "requiere": Un concepto necesita otro para funcionar
            - "mejora": Un concepto mejora el rendimiento de otro
            - "se_aplica_con": Conceptos que se usan juntos
            
            EJEMPLOS DE CONCEPTOS BIEN ESTRUCTURADOS:
            {{
                "nombre": "Few-shot Learning",
                "definicion": "Técnica que proporciona algunos ejemplos al modelo antes de la tarea real",
                "categoria": "tecnica",
                "importancia": 4,
                "ejemplos": ["Dar 3 ejemplos de traducción antes de traducir nueva frase"]
            }}
            
            {{
                "nombre": "LangChain",
                "definicion": "Framework para desarrollar aplicaciones con LLMs",
                "categoria": "herramienta", 
                "importancia": 5,
                "ejemplos": ["Cadenas de prompts", "Integraciones con APIs"]
            }}"""),
            ("human", """TRANSCRIPCIÓN A ANALIZAR:
            {transcripcion}
            
            RAZONA PASO A PASO:
            1. Primero, identifica TODOS los conceptos mencionados relacionados con prompt engineering, IA, y tecnologías relacionadas
            2. Para cada concepto, determina su categoría y nivel de importancia
            3. Identifica las relaciones explicadas o implícitas entre conceptos
            4. Asegúrate de capturar conceptos como: Prompt Engineering, LangChain, Gemini, Pydantic, Few-shot, Zero-shot, Chain of Thought, Structured Output, Tool Calling, etc.
            
            GENERA EL GRAFO DE CONOCIMIENTO COMPLETO EN FORMATO JSON.""")
        ])
    
    def extract_concepts(self, transcripcion: str) -> GrafoConocimiento:
        """Extract concepts using advanced prompt engineering techniques."""
        print("🧠 Extracting concepts using Chain of Thought...")
        
        prompt = self.create_analysis_prompt()
        chain = prompt | self.structured_llm
        
        result = chain.invoke({"transcripcion": transcripcion})
        print(f"✅ Extracted {len(result.conceptos)} concepts and {len(result.relaciones)} relationships!")
        
        return result
    
    def create_networkx_graph(self, grafo_conocimiento: GrafoConocimiento) -> nx.Graph:
        """Create NetworkX graph from knowledge graph data."""
        print("📊 Creating NetworkX graph...")
        
        G = nx.Graph()
        
        # Define color mapping for categories
        color_map = {
            "tecnica": "#FF6B6B",      # Red
            "herramienta": "#4ECDC4",   # Teal
            "metodologia": "#45B7D1",   # Blue
            "concepto": "#96CEB4"       # Green
        }
        
        # Add nodes with attributes
        for concepto in grafo_conocimiento.conceptos:
            G.add_node(
                concepto.nombre,
                categoria=concepto.categoria,
                importancia=concepto.importancia,
                definicion=concepto.definicion,
                ejemplos=concepto.ejemplos,
                color=color_map.get(concepto.categoria, "#CCCCCC"),
                size=concepto.importancia * 100  # Size based on importance
            )
        
        # Add edges with attributes
        for relacion in grafo_conocimiento.relaciones:
            if G.has_node(relacion.concepto_origen) and G.has_node(relacion.concepto_destino):
                G.add_edge(
                    relacion.concepto_origen,
                    relacion.concepto_destino,
                    peso=relacion.fuerza,
                    tipo=relacion.tipo_relacion,
                    width=relacion.fuerza * 5  # Width based on strength
                )
        
        print(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges!")
        return G
    
    def visualize_with_matplotlib(self, G: nx.Graph, grafo_conocimiento: GrafoConocimiento):
        """Create matplotlib visualization."""
        print("🎨 Creating matplotlib visualization...")
        
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better node distribution
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes by category
        for categoria, color in [("tecnica", "#FF6B6B"), ("herramienta", "#4ECDC4"), 
                               ("metodologia", "#45B7D1"), ("concepto", "#96CEB4")]:
            nodes = [node for node in G.nodes() if G.nodes[node].get('categoria') == categoria]
            if nodes:
                node_sizes = [G.nodes[node].get('size', 300) for node in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=color, node_size=node_sizes,
                                     alpha=0.8, label=categoria.title())
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v].get('width', 1) for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f"Grafo de Conocimiento: {grafo_conocimiento.tema_principal}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('knowledge_graph_matplotlib.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Matplotlib visualization saved as 'knowledge_graph_matplotlib.png'")
    
    def visualize_with_plotly(self, G: nx.Graph, grafo_conocimiento: GrafoConocimiento):
        """Create interactive Plotly visualization."""
        print("🎨 Creating interactive Plotly visualization...")
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            relacion_info = f"{edge[0]} → {edge[1]}<br>Tipo: {G[edge[0]][edge[1]].get('tipo', 'N/A')}<br>Fuerza: {G[edge[0]][edge[1]].get('peso', 0):.2f}"
            edge_info.append(relacion_info)
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Prepare node traces by category
        traces = [edge_trace]
        
        for categoria, color in [("tecnica", "#FF6B6B"), ("herramienta", "#4ECDC4"), 
                               ("metodologia", "#45B7D1"), ("concepto", "#96CEB4")]:
            nodes = [node for node in G.nodes() if G.nodes[node].get('categoria') == categoria]
            if not nodes:
                continue
                
            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            node_size = [G.nodes[node].get('importancia', 1) * 10 for node in nodes]
            
            node_text = []
            for node in nodes:
                node_info = G.nodes[node]
                adjacencies = list(G.neighbors(node))
                node_text.append(f"<b>{node}</b><br>" +
                               f"Categoría: {node_info.get('categoria', 'N/A')}<br>" +
                               f"Importancia: {node_info.get('importancia', 'N/A')}<br>" +
                               f"Conexiones: {len(adjacencies)}<br>" +
                               f"Definición: {node_info.get('definicion', 'N/A')[:100]}...")
            
            node_trace = go.Scatter(x=node_x, y=node_y,
                                   mode='markers+text',
                                   hoverinfo='text',
                                   hovertext=node_text,
                                   text=nodes,
                                   textposition="middle center",
                                   marker=dict(size=node_size,
                                             color=color,
                                             line=dict(width=2, color='white')),
                                   name=categoria.title())
            traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title=dict(
                               text=f'<b>Grafo de Conocimiento Interactivo: {grafo_conocimiento.tema_principal}</b>',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Resumen: {grafo_conocimiento.resumen[:200]}...",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=10)
                           ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'))
        
        fig.write_html('knowledge_graph_interactive.html')
        print("✅ Interactive Plotly visualization saved as 'knowledge_graph_interactive.html'")
        
        # Optional: Try to show the plot with timeout (disabled by default to avoid hanging)
        show_in_browser = False  # Set to True if you want to try opening in browser
        
        if show_in_browser:
            try:
                print("🌐 Attempting to open plot in browser...")
                fig.show()
                print("✅ Interactive plot opened in browser")
            except Exception as e:
                print(f"⚠️  Could not open plot in browser: {e}")
                print("📄 You can open 'knowledge_graph_interactive.html' manually in your browser")
        else:
            print("📄 To view the interactive plot, open 'knowledge_graph_interactive.html' in your browser")
    
    def save_results(self, grafo_conocimiento: GrafoConocimiento, transcripcion: str):
        """Save all results to files."""
        print("💾 Saving results...")
        
        # Save structured data as JSON
        with open('knowledge_graph_data.json', 'w', encoding='utf-8') as f:
            json.dump(grafo_conocimiento.dict(), f, ensure_ascii=False, indent=2)
        
        # Save transcription (already saved in transcribe_audio method, but keeping for completeness)
        with open('transcription.txt', 'w', encoding='utf-8') as f:
            f.write(transcripcion)
        
        # Save summary report
        with open('knowledge_graph_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# Reporte del Grafo de Conocimiento\n\n")
            f.write(f"**Tema Principal:** {grafo_conocimiento.tema_principal}\n\n")
            f.write(f"**Resumen:** {grafo_conocimiento.resumen}\n\n")
            f.write(f"## Estadísticas\n")
            f.write(f"- **Total de Conceptos:** {len(grafo_conocimiento.conceptos)}\n")
            f.write(f"- **Total de Relaciones:** {len(grafo_conocimiento.relaciones)}\n\n")
            
            # Concepts by category
            categorias = {}
            for concepto in grafo_conocimiento.conceptos:
                if concepto.categoria not in categorias:
                    categorias[concepto.categoria] = []
                categorias[concepto.categoria].append(concepto)
            
            f.write(f"## Conceptos por Categoría\n\n")
            for categoria, conceptos in categorias.items():
                f.write(f"### {categoria.title()} ({len(conceptos)} conceptos)\n")
                for concepto in sorted(conceptos, key=lambda x: x.importancia, reverse=True):
                    f.write(f"- **{concepto.nombre}** (Importancia: {concepto.importancia}): {concepto.definicion}\n")
                f.write("\n")
        
        print("✅ Results saved to files!")
    
    def process_complete_workflow(self, audio_file_path: str):
        """Execute the complete workflow."""
        print("🚀 Starting complete knowledge graph creation workflow...\n")
        
        # Step 1: Transcribe audio
        transcripcion = self.transcribe_audio(audio_file_path)
        
        # Step 2: Extract concepts and relationships
        grafo_conocimiento = self.extract_concepts(transcripcion)
        
        # Step 3: Create graph
        G = self.create_networkx_graph(grafo_conocimiento)
        
        # Step 4: Create visualizations
        self.visualize_with_matplotlib(G, grafo_conocimiento)
        self.visualize_with_plotly(G, grafo_conocimiento)
        
        # Step 5: Save results
        self.save_results(grafo_conocimiento, transcripcion)
        
        print(f"\n🎉 Workflow completed successfully!")
        print(f"📊 Created knowledge graph with {len(grafo_conocimiento.conceptos)} concepts")
        print(f"🔗 Found {len(grafo_conocimiento.relaciones)} relationships")
        print(f"📁 All results saved to current directory")
        
        return grafo_conocimiento, G

# Example usage
if __name__ == "__main__":
    creator = KnowledgeGraphCreator()
    
    # Process the audio file
    audio_file = "2025-06-02_15-06-35_audio.mp3"
    
    if os.path.exists(audio_file):
        grafo, graph = creator.process_complete_workflow(audio_file)
        
        # Print summary
        print(f"\n📋 RESUMEN FINAL:")
        print(f"Tema: {grafo.tema_principal}")
        print(f"Conceptos encontrados: {len(grafo.conceptos)}")
        print(f"Relaciones identificadas: {len(grafo.relaciones)}")
    else:
        print(f"❌ Audio file '{audio_file}' not found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith(('.mp3', '.wav', '.m4a')):
                print(f"  - {file}") 