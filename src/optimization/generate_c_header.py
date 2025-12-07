import numpy as np
from pathlib import Path
import argparse

def convert_tflite_to_c_header(tflite_path: Path, output_path: Path, array_name: str = "model"):
    """Convert TFLite model to C header file for embedding in firmware."""
    
    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    
    # Generate C header content
    header_content = f"""// Auto-generated C header file
// Model: {tflite_path.name}
// Size: {model_size} bytes ({model_size/1024:.2f} KB)
// Generated: {Path(__file__).name}

#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

// Model size in bytes
#define {array_name.upper()}_SIZE {model_size}

// Model data array
alignas(8) const unsigned char {array_name}_data[] = {{
"""
    
    # Add model bytes (16 bytes per line)
    for i in range(0, len(model_data), 16):
        chunk = model_data[i:i+16]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        header_content += f"  {hex_values},\n"
    
    # Remove trailing comma
    header_content = header_content.rstrip(',\n') + '\n'
    
    header_content += f"""
}};

#endif  // {array_name.upper()}_H
"""
    
    # Write header file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"C header generated: {output_path}")
    print(f"Model size: {model_size} bytes ({model_size/1024:.2f} KB)")
    print(f"Array name: {array_name}_data")
    print(f"\nUsage in Arduino/C++:")
    print(f'  #include "{output_path.name}"')
    print(f'  const tflite::Model* model = tflite::GetModel({array_name}_data);')

def main():
    parser = argparse.ArgumentParser(description="Convert TFLite model to C header")
    parser.add_argument("--input", type=str, default="data/models/audio_classifier_int8.tflite")
    parser.add_argument("--output", type=str, default="data/models/model.h")
    parser.add_argument("--array-name", type=str, default="audio_model")
    
    args = parser.parse_args()
    
    convert_tflite_to_c_header(
        Path(args.input),
        Path(args.output),
        args.array_name
    )

if __name__ == "__main__":
    main()
