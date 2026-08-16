import os
from PIL import Image, ImageDraw
from agent import DocumentIntelligenceAgent

def create_sample_invoice(output_path: str = "sample_invoice.png"):
    """Genera una imagen simple de factura para pruebas de OCR si no existe una."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Texto simulando el encabezado y contenido de una factura
    lines = [
        (40, 40, "TECH SOLUTIONS INC."),
        (40, 100, "Invoice Number: INV-2026-8891"),
        (40, 140, "Invoice Date: 2026-08-15"),
        (40, 200, "Description: Cloud Architecture Advisory"),
        (40, 260, "Total Amount Due: $1,450.00"),
    ]

    for x, y, text in lines:
        draw.text((x, y), text, fill=(0, 0, 0))

    img.save(output_path)
    return output_path

if __name__ == "__main__":
    image_path = "sample_invoice.png"
    
    # 1. Crear la imagen de prueba si no existe
    if not os.path.exists(image_path):
        print(f"Generando imagen de factura de prueba: {image_path}")
        create_sample_invoice(image_path)

    # 2. Instanciar el agente
    agent = DocumentIntelligenceAgent(confidence_threshold=50)

    # 3. Cargar la imagen y extraer tokens espaciales vía OCR
    print("Ejecutando OCR y análisis espacial...")
    image = Image.open(image_path)
    tokens = agent.extract_tokens(image)
    print(f"Tokens detectados: {len(tokens)}")

    # 4. Extraer campos estructurados guiados por el esquema
    extracted_data = agent.extract_fields_from_tokens(tokens)

    # 5. Mostrar resultados estructurados
    print("\n--- Campos Extraídos del Documento ---")
    for field, data in extracted_data.items():
        value = data["value"]
        conf = data["confidence"]
        bbox = data["provenance"]["bbox"] if data["provenance"] else "N/A"
        print(f"\nCampo: {field.upper()}")
        print(f"  • Valor extraído: {value}")
        print(f"  • Confianza: {conf}%")
        print(f"  • Bounding Box: {bbox}")