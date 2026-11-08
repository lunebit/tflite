# TensorFlow Lite Bindings for V

TensorFlow Lite bindings for the V programming language with type-safe API and zero-copy tensor operations. 
All library functions are well documented.

## Features

-  **Type-safe API** with compile-time type checking
-  **Zero-copy tensor operations** for maximum performance
-  **High-level ModelManager** for easy inference
-  **Low-level API** for advanced use cases
-  **XNNPACK delegate** support for optimized CPU inference
-  **Cross-platform** (macOS, Linux, Windows)
-  **Comprehensive documentation** with examples
-  **Memory-safe** with RAII pattern and proper resource management

## Requirements

- V compiler (0.5 or later)
- TensorFlow Lite C library
- C compiler (GCC, Clang, or MSVC)

## Installation

### 1. Install the module

```bash
v install https://github.com/lunebit/tflite
```

Or clone directly (better) inside your project:
```bash
git clone https://github.com/lunebit/tflite.git
```

### 2. Install TensorFlow Lite

**MacOS Install Bazelisk**
```bash
brew install bazelisk
```

**Linux / Macos (from source):**
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
bazel build -c opt //tensorflow/lite/c:tensorflowlite_c \
  --define tflite_with_xnnpack=true \
  //tensorflow/lite/c:tensorflowlite_c
```

**MacOS**
```bash
bazel build -c opt \
  --config=macos_arm64 \
  --define tflite_with_xnnpack=true \
  //tensorflow/lite/c:tensorflowlite_c
```
**Put the library to tflite/lib dirrectory:**
```bash
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib /yourpath/tflite/lib/mac
```

Notice: For MacOS users you can download precompiled tensorflow lite library (m silicon) from here: [https://github.com/esrsound/assets/tflite](https://github.com/esrsound/tflite/blob/main/lib/mac/libtensorflowlite_c.dylib)


## Quick Start

```v
import tflite

fn main() {
    // Create model manager
    mut manager := tflite.new_model_manager()
    defer { manager.free() }
    
    // Load model
    manager.load_model('model.tflite', 'my_model', 4, true)!
    
    // Prepare input data
    input_data := [f32(1.0), 2.0, 3.0, 4.0]
    manager.set_input(0, input_data)!
    
    // Run inference
    result := manager.run_inference()!
    
    // Get output (safe copy)
    output := result.copy_output[f32](0)!
    println('Output: ${output}')
}
```

## Configuration

### Custom Library Paths

You can specify custom paths for TensorFlow Lite headers and libraries:

```bash
v -d tflite_include_dir="/path/to/tensorflow" \
  -d tflite_lib_dir="/path/to/libs" \
  -d tflite_lib_name="tensorflowlite_c" \
  run main.v
```

**Default paths:**
- **macOS:** `/usr/local/include`, `@VMODROOT/tflite`, `tensorflowlite_c-mac`
- **Linux:** `/usr/local/include`, `/usr/local/lib`, `tensorflowlite_c`

### Build Script Example

Create `build.sh`:
```bash
#!/bin/bash
v -d tflite_include_dir="/usr/local/include" \
  -d tflite_lib_dir="./tflite" \
  run main.v -prod -gc boehm
```

## API Documentation

### High-Level API (Recommended)

#### ModelManager

```v
// Create manager
mut manager := tflite.new_model_manager()
defer { manager.free() }

// Load model with XNNPACK acceleration
manager.load_model('model.tflite', 'model_name', num_threads: 4, xnnpack: true)!

// Set input data (type-safe)
manager.set_input[f32](0, input_data)!

// Run inference
result := manager.run_inference()!

// Get output (safe copy - data survives after manager.free())
output := result.copy_output[f32](0)!
```

### Low-Level API (Advanced)

```v
// Create model
model := tflite.create_model_from_file('model.tflite')!
defer { model.free() }

// Create interpreter options
mut options := tflite.create_interpreter_options()
options.set_num_threads(4)
options.enable_xnnpack(4)!

// Create interpreter
mut interpreter := tflite.create_interpreter_with_options(model, mut options)!
defer { interpreter.free() }

// Allocate tensors
interpreter.allocate_tensors()!

// Get input tensor
input_tensor := interpreter.get_input_tensor(0)!
input_tensor.copy_from[f32](input_data)!

// Run inference
interpreter.invoke()!

// Get output tensor
output_tensor := interpreter.get_output_tensor(0)!
mut output_data := []f32{len: 10}
output_tensor.copy_to[f32](mut output_data)!
```

### Supported Types

The library supports type-safe operations with the following types:
- `f32` (float32)
- `f64` (float64)
- `i8`, `i16`, `i32`, `i64` (signed integers)
- `u8`, `u16`, `u32`, `u64` (unsigned integers)
- `bool`

### Zero-Copy vs Safe-Copy

**⚠️ Zero-copy (advanced users only):**
```v
output := result.get_output[f32](0)!  // Points to C memory
// ⚠️ Invalid after manager.free()!
```

**✅ Safe-copy (recommended):**
```v
output := result.copy_output[f32](0)!  // Copies data
manager.free()  // Safe - data still valid
```

## Performance Tips

1. **Enable XNNPACK** for CPU optimization:
   ```v
   manager.load_model('model.tflite', 'model', 4, true)!  // xnnpack: true
   ```

2. **Use zero-copy for hot paths** (if you know what you're doing):
   ```v
   output := result.get_output[f32](0)!  // No memory copy
   process(output)
   // Don't call manager.free() while using output!
   ```
3. **Set multi-thread option:**
4. ```v
    manager.set_num_threads(4)
    ```

## Error Handling

All operations return `Result` types with detailed error messages:

```v
manager.load_model('model.tflite', 'model', 4, false) or {
    eprintln('Failed to load model: ${err}')
    return
}
```

## Architecture

```
High-Level API (ModelManager)
    ↓
Mid-Level API (Interpreter, Model, Options)
    ↓
Low-Level API (Tensor operations)
    ↓
TensorFlow Lite C API
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow Lite](https://www.tensorflow.org/lite) team for the excellent C API
- [V language](https://vlang.io) community

## Support

- 🐛 [Issue Tracker](https://github.com/esrsound/tflite/issues)
- 💬 [Discussions](https://github.com/esrsound/tflite/discussions)

***
