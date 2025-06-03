#!/usr/bin/env python3
"""
Simple Demo Script for Knowledge Graph Creation
Based on the blog: https://www.philschmid.de/gemini-langchain-cheatsheet
"""

import os
import json
from knowledge_graph_creator import KnowledgeGraphCreator

def demo_with_sample_text():
    """Demo using sample educational text instead of audio."""
    
    print("🚀 Knowledge Graph Demo - Sample Text Version")
    print("=" * 50)
    
    # Sample educational content about prompt engineering
    sample_content = """
    En esta sesión de Prompt Engineering hemos cubierto varios conceptos fundamentales.
    
    Few-shot Learning es una técnica donde proporcionamos algunos ejemplos al modelo 
    antes de pedirle que realice la tarea principal. Por ejemplo, si queremos que 
    traduzca, podemos dar 2-3 ejemplos de traducción antes.
    
    Zero-shot Learning, por el contrario, significa que el modelo debe realizar la 
    tarea sin ejemplos previos, basándose únicamente en instrucciones claras y precisas.
    
    Chain of Thought es una técnica poderosa que permite al modelo razonar paso a paso.
    Le pedimos que "piense en voz alta" y muestre su proceso de razonamiento.
    
    LangChain es un framework crucial para desarrollar aplicaciones con LLMs. Nos 
    permite encadenar operaciones, integrar con APIs como Gemini, y estructurar 
    nuestros prompts de manera profesional.
    
    Gemini API es la interfaz de Google para acceder a sus modelos avanzados. 
    Soporta texto, imágenes, audio y video.
    
    Pydantic nos ayuda a estructurar las salidas en formato JSON de manera consistente
    y validada. Es esencial para obtener datos estructurados confiables.
    
    Los embeddings son representaciones vectoriales de texto que permiten búsquedas
    semánticas. Se usan en RAG (Retrieval Augmented Generation).
    
    Tool Calling permite que los modelos llamen funciones externas para obtener
    información actualizada o realizar acciones específicas.
    
    NetworkX es una biblioteca de Python para crear, manipular y estudiar la 
    estructura de grafos complejos.
    """
    
    try:
        # Initialize the creator
        creator = KnowledgeGraphCreator()
        print("✅ Knowledge Graph Creator initialized")
        
        # Extract concepts from sample text
        print("\n🧠 Extracting concepts using advanced prompt engineering...")
        grafo_conocimiento = creator.extract_concepts(sample_content)
        
        # Show results
        print(f"\n📊 RESULTS:")
        print(f"  • Concepts found: {len(grafo_conocimiento.conceptos)}")
        print(f"  • Relationships identified: {len(grafo_conocimiento.relaciones)}")
        print(f"  • Main topic: {grafo_conocimiento.tema_principal}")
        
        # Create graph
        print("\n📈 Creating NetworkX graph...")
        G = creator.create_networkx_graph(grafo_conocimiento)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        creator.visualize_with_matplotlib(G, grafo_conocimiento)
        creator.visualize_with_plotly(G, grafo_conocimiento)
        
        # Save results
        print("\n💾 Saving results...")
        creator.save_results(grafo_conocimiento, sample_content)
        
        # Show extracted concepts by category
        print(f"\n🔍 CONCEPTS BY CATEGORY:")
        print("=" * 30)
        
        categorias = {}
        for concepto in grafo_conocimiento.conceptos:
            if concepto.categoria not in categorias:
                categorias[concepto.categoria] = []
            categorias[concepto.categoria].append(concepto)
        
        for categoria, conceptos in categorias.items():
            print(f"\n📚 {categoria.upper()} ({len(conceptos)} concepts):")
            for concepto in sorted(conceptos, key=lambda x: x.importancia, reverse=True):
                print(f"  • {concepto.nombre} (⭐{concepto.importancia})")
                print(f"    {concepto.definicion[:60]}...")
        
        print(f"\n🔗 SAMPLE RELATIONSHIPS:")
        for i, relacion in enumerate(grafo_conocimiento.relaciones[:5]):
            print(f"  {i+1}. {relacion.concepto_origen} --[{relacion.tipo_relacion}]--> {relacion.concepto_destino}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Check these files:")
        print(f"  • knowledge_graph_data.json - Structured data")
        print(f"  • knowledge_graph_report.md - Detailed report")
        print(f"  • knowledge_graph_matplotlib.png - Static visualization")
        print(f"  • knowledge_graph_interactive.html - Interactive visualization")
        
        return grafo_conocimiento, G
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("  1. Google AI API key set as GOOGLE_API_KEY environment variable")
        print("  2. All required packages installed (pip install -r requirements.txt)")
        return None, None

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'langchain_google_genai',
        'pydantic', 
        'networkx',
        'matplotlib',
        'plotly'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"  • {pkg}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    print("🔍 Checking requirements...")
    
    if not check_requirements():
        return
    
    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not found in environment variables")
        api_key = input("Enter your Google AI API key: ").strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            print("❌ API key is required to run the demo")
            return
    
    # Run demo
    demo_with_sample_text()

if __name__ == "__main__":
    main() 