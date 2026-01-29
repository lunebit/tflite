// TensorFlow Lite C API bindings for V
// This module provides bindings for TensorFlow Lite inference using the C API
//
// Configuration:
//   To customize paths, use compiler flags when compiling:
//   -d tflite_include_dir: Path to TensorFlow Lite headers
//   -d tflite_lib_dir: Path to TensorFlow Lite libraries
//   -d tflite_lib_name: Library name (default: tensorflowlite_c-mac on macOS, tensorflowlite_c on Linux)
//
// Example:
//   v -d tflite_include_dir=/path/to/tensorflow -d tflite_lib_dir=/path/to/libs your_app.v

module tflite

// ============================================================================
// Compiler flags and library paths
// ============================================================================

$if macos {
	// flag to tensorflow lite library include paths and library paths on macOS
	#flag -I $d('tflite_include_dir','/usr/local/include')
	#flag -L $d('tflite_lib_dir', '@VMODROOT/lib/mac')
	#flag -Wl,-rpath,$d('tflite_lib_dir', '@VMODROOT/lib/mac')
	#flag -l $d('tflite_lib_name', 'tensorflowlite_c')
} $else $if linux {
	// flag to tensorflow lite library include paths and library paths on Linux
	#flag -I $d('tflite_include_dir','/usr/local/include')
	#flag -L $d('tflite_lib_dir', '@VMODROOT/lib')
	#flag -l $d('tflite_lib_name', 'tensorflowlite_c')
} $else $if windows {
	// Other platforms (Windows, etc.)
	#flag -I $d('tflite_include_dir','C:/tensorflow/include')
	#flag -L $d('tflite_lib_dir', 'C:/tensorflow/lib')
	#flag -l $d('tflite_lib_name', 'tensorflowlite_c')
}

// Include TensorFlow Lite C API headers
#include <tensorflow/lite/c/c_api.h> 
// XNNPACK delegate header
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

// ============================================================================
// Status codes
// ============================================================================

pub enum TfLiteStatus as int {
	ok = 0
	error = 1
	delegate_error = 2
	application_error = 3
	delegate_data_not_found = 4
	delegate_data_write_error = 5
	delegate_data_read_error = 6
	unresolved_ops = 7
	cancelled = 8
	output_shape_not_known = 9
}

// ============================================================================
// Tensor types
// ============================================================================

pub enum TfLiteType as int {
	no_type = 0
	float32 = 1
	int32 = 2
	uint8 = 3
	int64 = 4
	string = 5
	bool = 6
	int16 = 7
	complex64 = 8
	int8 = 9
	float16 = 10
	float64 = 11
	complex128 = 12
	uint64 = 13
	resource = 14
	variant = 15
	uint32 = 16
	uint16 = 17
	int4 = 18
	bfloat16 = 19
	int2 = 20
}

// ============================================================================
// Union Types for zero-copy tensor access
// ============================================================================

pub struct TypedTensorData {
pub:
	tensor_type TfLiteType
	data_ptr    voidptr
	size        int
}

// matches_type checks if a V type matches a TensorFlow Lite tensor type.
// This is a generic function for type compatibility checking.
//
// Parameters:
//   T - V data type (e.g., f32, i32, u8, etc.)
//   tensor_type - TensorFlow Lite tensor type to match against
//
// Returns:
//   true if types match, false otherwise
//
// Supported type mappings:
//   - f32 -> TfLiteType.float32
//   - f64 -> TfLiteType.float64
//   - i32 -> TfLiteType.int32
//   - i64 -> TfLiteType.int64
//   - u8  -> TfLiteType.uint8
//   - i8  -> TfLiteType.int8
//   - u16 -> TfLiteType.uint16
//   - i16 -> TfLiteType.int16
//   - u32 -> TfLiteType.uint32
//   - u64 -> TfLiteType.uint64
//   - bool -> TfLiteType.bool
//
// Note: Some TensorFlow Lite types (float16, complex64, complex128, int4, int2,
// bfloat16, resource, variant, string) are not directly supported by V's type system
// and require special handling.
pub fn matches_type[T](tensor_type TfLiteType) bool {
	$if T is f32 {
		return tensor_type == .float32
	} $else $if T is f64 {
		return tensor_type == .float64
	} $else $if T is i32 {
		return tensor_type == .int32
	} $else $if T is i64 {
		return tensor_type == .int64
	} $else $if T is u8 {
		return tensor_type == .uint8
	} $else $if T is i8 {
		return tensor_type == .int8
	} $else $if T is u16 {
		return tensor_type == .uint16
	} $else $if T is i16 {
		return tensor_type == .int16
	} $else $if T is u32 {
		return tensor_type == .uint32
	} $else $if T is u64 {
		return tensor_type == .uint64
	} $else $if T is bool {
		return tensor_type == .bool
	} $else {
		return false
	}
}

// get_type_size returns the size in bytes of a TensorFlow Lite tensor type.
// This is useful for calculating buffer sizes and element counts.
//
// Parameters:
//   tensor_type - TensorFlow Lite tensor type
//
// Returns:
//   Size in bytes, or 0 for unknown/unsupported types
//
// Supported types:
//   - float32, float64, int32, int64, uint8, int8, uint16, int16, uint32, uint64, bool
//   - float16 (2 bytes)
//   - complex64 (8 bytes)
//   - complex128 (16 bytes)
//   - int4, int2 (0.5, 0.25 bytes - not directly supported)
//   - bfloat16 (2 bytes)
//   - resource, variant, string (variable size)
pub fn get_type_size(tensor_type TfLiteType) int {
	match tensor_type {
		.float32 { return 4 }
		.float64 { return 8 }
		.int32 { return 4 }
		.int64 { return 8 }
		.uint8 { return 1 }
		.int8 { return 1 }
		.uint16 { return 2 }
		.int16 { return 2 }
		.uint32 { return 4 }
		.uint64 { return 8 }
		.bool { return 1 }
		.float16 { return 2 }
		.complex64 { return 8 }
		.complex128 { return 16 }
		.bfloat16 { return 2 }
		.int4 { return 1 } // Packed: 2 int4 values per byte
		.int2 { return 1 } // Packed: 4 int2 values per byte
		.string { return 0 } // Variable size
		.resource { return 0 } // Variable size
		.variant { return 0 } // Variable size
		else { return 0 }
	}
}

// matches_type checks if a V type matches the tensor type of TypedTensorData.
// Parameters:
//  T - V data type (e.g., f32, i32, u8, etc.)
// Returns:
//	 true if types match, false otherwise

pub fn (t &TypedTensorData) matches_type[T]() bool {
	return matches_type[T](t.tensor_type)
}
// matches_type checks if a V type matches the tensor type of Tensor.
pub fn (t &Tensor) matches_type[T]() bool {
	tensor_type := t.get_type()
	return matches_type[T](tensor_type)
}


// get_data returns a zero-copy slice pointing to the tensor's C-managed memory.
//
// ⚠️ CRITICAL WARNING: The returned slice points directly to C-managed memory.
// The slice becomes INVALID (dangling pointer) if:
// - The underlying ModelManager is freed
// - The model is unloaded
// - The interpreter is freed
//
// For safe usage, either:
// 1. Copy the data immediately using C.memcpy() or copy_output[T]()
// 2. Ensure the C object outlives the slice (keep ModelManager alive)
//
// Example of safe usage:
// ```v
// mut manager := new_model_manager()
// manager.load_model("model.tflite", "test", 1, false)!
// manager.set_input(0, input_data)!
// result := manager.run_inference()!
//
// // SAFE: Copy data while manager is still alive
// output_data := result.copy_output[f32](0)!
//
// // Now we can safely free the manager
// manager.free()
//
// // output_data is still valid (it's a copy)
// ```
//
// Example of UNSAFE usage (DO NOT DO THIS):
// ```v
// mut output_data := []f32{}
// {
//     mut manager := new_model_manager()
//     manager.load_model("model.tflite", "test", 1, false)!
//     manager.set_input(0, input_data)!
//     result := manager.run_inference()!
//
//     // UNSAFE: Zero-copy slice
//     output_data = result.get_output[f32](0)!
//
//     manager.free()  // manager is freed here
// }  // output_data is now INVALID - dangling pointer!
//
// // Using output_data here would cause undefined behavior!
// ```
pub fn (t &TypedTensorData) get_data[T]() ![]T {
	if !t.matches_type[T]() {
        return error('Requested type does not match tensor type ${t.tensor_type}')
    }
	unsafe {
		mut arr := []T{len: 0}
		arr.data = t.get_ptr()
		arr.len = t.size
		arr.cap = t.size
		return arr
	}
}

// get_ptr returns the raw pointer to the tensor data.
// This is an internal helper function.
fn (t &TypedTensorData) get_ptr() voidptr {
    return t.data_ptr
}
// ============================================================================
// Quantization parameters
// ============================================================================

pub struct C.TfLiteQuantizationParams {
	scale      f32
	zero_point i32
}

// ============================================================================
// Opaque types
// ============================================================================

@[typedef]
pub struct C.TfLiteModel {}

@[typedef]
pub struct C.TfLiteInterpreterOptions {}

@[typedef]
pub struct C.TfLiteInterpreter {}

@[typedef]
pub struct C.TfLiteTensor {}

@[typedef]
pub struct C.TfLiteDelegate {}

@[typedef]
pub struct C.TfLiteXNNPackDelegateOptions {
	num_threads i32
	flags       u32
}


// ============================================================================
// C function declarations
// ============================================================================

fn C.TfLiteVersion() &char

fn C.TfLiteModelCreate(voidptr, usize) &C.TfLiteModel
fn C.TfLiteModelCreateFromFile(&char) &C.TfLiteModel
fn C.TfLiteModelDelete(&C.TfLiteModel)

fn C.TfLiteInterpreterOptionsCreate() &C.TfLiteInterpreterOptions
fn C.TfLiteInterpreterOptionsCopy(&C.TfLiteInterpreterOptions) &C.TfLiteInterpreterOptions
fn C.TfLiteInterpreterOptionsDelete(&C.TfLiteInterpreterOptions)
fn C.TfLiteInterpreterOptionsSetNumThreads(&C.TfLiteInterpreterOptions, i32)

fn C.TfLiteXNNPackDelegateOptionsDefault() C.TfLiteXNNPackDelegateOptions
fn C.TfLiteXNNPackDelegateCreate(&C.TfLiteXNNPackDelegateOptions) &C.TfLiteDelegate
fn C.TfLiteXNNPackDelegateDelete(&C.TfLiteDelegate)
fn C.TfLiteInterpreterOptionsAddDelegate(&C.TfLiteInterpreterOptions, &C.TfLiteDelegate)

fn C.TfLiteInterpreterCreate(&C.TfLiteModel, &C.TfLiteInterpreterOptions) &C.TfLiteInterpreter
fn C.TfLiteInterpreterDelete(&C.TfLiteInterpreter)
fn C.TfLiteInterpreterGetInputTensorCount(&C.TfLiteInterpreter) i32
fn C.TfLiteInterpreterGetInputTensor(&C.TfLiteInterpreter, i32) &C.TfLiteTensor
fn C.TfLiteInterpreterResizeInputTensor(&C.TfLiteInterpreter, i32, &int, i32) TfLiteStatus
fn C.TfLiteInterpreterAllocateTensors(&C.TfLiteInterpreter) TfLiteStatus
fn C.TfLiteInterpreterInvoke(&C.TfLiteInterpreter) TfLiteStatus
fn C.TfLiteInterpreterGetOutputTensorCount(&C.TfLiteInterpreter) i32
fn C.TfLiteInterpreterGetOutputTensor(&C.TfLiteInterpreter, i32) &C.TfLiteTensor

fn C.TfLiteTensorType(&C.TfLiteTensor) TfLiteType
fn C.TfLiteTensorNumDims(&C.TfLiteTensor) i32
fn C.TfLiteTensorDim(&C.TfLiteTensor, i32) i32
fn C.TfLiteTensorByteSize(&C.TfLiteTensor) usize
fn C.TfLiteTensorData(&C.TfLiteTensor) voidptr
fn C.TfLiteTensorName(&C.TfLiteTensor) &char
fn C.TfLiteTensorQuantizationParams(&C.TfLiteTensor) C.TfLiteQuantizationParams
fn C.TfLiteTensorCopyFromBuffer(&C.TfLiteTensor, voidptr, usize) TfLiteStatus
fn C.TfLiteTensorCopyToBuffer(&C.TfLiteTensor, voidptr, usize) TfLiteStatus

// ============================================================================
// High-level V wrapper types
// ============================================================================

@[heap]
pub struct Model {
mut:
	c_model &C.TfLiteModel
}
@[heap]
pub struct InterpreterOptions {
mut:
	c_options        &C.TfLiteInterpreterOptions
	xnnpack_delegate &C.TfLiteDelegate
}
@[heap]
pub struct Interpreter {
mut:
	c_interpreter &C.TfLiteInterpreter
	options       InterpreterOptions // Interpreter owns options (RAII)
}
@[heap]
pub struct Tensor {
	c_tensor &C.TfLiteTensor
}

// ============================================================================
// Model methods
// ============================================================================

// create_model_from_file creates a TensorFlow Lite model from a file.
//
// Parameters:
//   path - path to the .tflite model file
//
// Returns:
//   Model instance
//
// Errors:
//   - If the file cannot be loaded
//
// Example:
// ```v
// model := create_model_from_file("model.tflite")!
// defer { model.free() }
// ```
pub fn create_model_from_file(path string) !Model {
	c_path := path.str
	c_model := C.TfLiteModelCreateFromFile(c_path)
	if c_model == 0 {
		return error('Failed to create model from file: ${path}')
	}
	return Model{c_model: c_model}
}

// create_model_from_buffer creates a TensorFlow Lite model from a byte buffer.
//
// Parameters:
//   data - byte buffer containing the model
//
// Returns:
//   Model instance
//
// Errors:
//   - If the buffer is empty
//   - If the model cannot be created
//
// Example:
// ```v
// model_data := os.read_file("model.tflite")!
// model := create_model_from_buffer(model_data)!
// defer { model.free() }
// ```
pub fn create_model_from_buffer(data []u8) !Model {
	if data.len == 0 {
		return error('Model buffer is empty')
	}
	c_model := C.TfLiteModelCreate(data.data, usize(data.len))
	if c_model == 0 {
		return error('Failed to create model from buffer')
	}
	return Model{c_model: c_model}
}

// free releases the TensorFlow Lite model resources.
// It is safe to call this method multiple times.
//
// Example:
// ```v
// model := create_model_from_file("model.tflite")!
// defer { model.free() }
// ```
pub fn (mut m Model) free() {
	if m.c_model != 0 {
		C.TfLiteModelDelete(m.c_model)
		m.c_model = unsafe { nil }
	}
}

// ============================================================================
// InterpreterOptions methods
// ============================================================================

// create_interpreter_options creates a new InterpreterOptions instance.
// The options are used to configure the interpreter before creation.
//
// Returns:
//   New InterpreterOptions instance
//
// Example:
// ```v
// mut options := create_interpreter_options()
// options.set_num_threads(4)
// options.enable_xnnpack(2) or { println('XNNPACK not available') }
// mut interpreter := create_interpreter_with_options(model, mut options)!
// ```
pub fn create_interpreter_options() InterpreterOptions {
	c_options := C.TfLiteInterpreterOptionsCreate()
	return InterpreterOptions{
		c_options: c_options
		xnnpack_delegate: unsafe { nil }
	}
}


// set_num_threads sets the number of threads for interpreter execution.
// This affects the parallelism of model execution.
// Example:
// ```v
// options := create_interpreter_options()
// options.set_num_threads(4)
// mut interpreter := create_interpreter_with_options(model, mut options)!
// ```
pub fn (o &InterpreterOptions) set_num_threads(num_threads int) {
	C.TfLiteInterpreterOptionsSetNumThreads(o.c_options, i32(num_threads))
}

// enable_xnnpack enables the XNNPACK delegate for optimized CPU inference.
//
// Parameters:
//   num_threads - number of threads to use for XNNPACK
//
// Errors:
//   - If XNNPACK is not supported on this platform
//   - If delegate creation fails
//
// Example:
// ```v
// mut options := create_interpreter_options()
// options.enable_xnnpack(2) or { println('XNNPACK not available') }
// ```
pub fn (mut o InterpreterOptions) enable_xnnpack(num_threads int) ! {

		mut xnn_options := C.TfLiteXNNPackDelegateOptionsDefault()
		xnn_options.num_threads = i32(num_threads)
		o.xnnpack_delegate = C.TfLiteXNNPackDelegateCreate(&xnn_options)
		
		if o.xnnpack_delegate == 0 {
			return error('Failed to create XNNPACK delegate')
		}
		C.TfLiteInterpreterOptionsAddDelegate(o.c_options, o.xnnpack_delegate)
}

// free releases the interpreter options resources.
// This is called automatically by Interpreter.free().
// It is safe to call this method multiple times.
pub fn (mut o InterpreterOptions) free() {

	if o.xnnpack_delegate != 0 {
		C.TfLiteXNNPackDelegateDelete(o.xnnpack_delegate)
		o.xnnpack_delegate = unsafe { nil }
	}
	if o.c_options != 0 {
		C.TfLiteInterpreterOptionsDelete(o.c_options)
		o.c_options = unsafe { nil }
	}
}

// ============================================================================
// Interpreter methods
// ============================================================================

// create_interpreter_with_options creates a TensorFlow Lite interpreter with the given options.
// The interpreter takes ownership of the options (RAII pattern).
//
// Parameters:
//   model - the loaded model
//   options - interpreter configuration options
//
// Returns:
//   Interpreter instance
//
// Errors:
//   - If interpreter creation fails
//
// Example:
// ```v
// mut options := create_interpreter_options()
// options.set_num_threads(4)
// mut interpreter := create_interpreter_with_options(model, mut options)!
// defer { interpreter.free() }
// ```
pub fn create_interpreter_with_options(model &Model, mut options InterpreterOptions) !Interpreter {
    c_interpreter := C.TfLiteInterpreterCreate(model.c_model, options.c_options)
    if c_interpreter == 0 {
        return error('Failed to create interpreter')
    }
    // Ownership of options is transferred to Interpreter (RAII)
    return Interpreter{
        c_interpreter: c_interpreter
        options: options
    }
}

// free releases the interpreter resources.
// This also frees the owned InterpreterOptions.
// It is safe to call this method multiple times.
//
// Example:
// ```v
// mut interpreter := create_interpreter_with_options(model, mut options)!
// defer { interpreter.free() }
// ```
pub fn (mut i Interpreter) free() {
	if i.c_interpreter != 0 {
		C.TfLiteInterpreterDelete(i.c_interpreter)
		i.c_interpreter = unsafe { nil }
	}
	// Free owned options (RAII)
	i.options.free()
}

// get_input_tensor_count returns the number of input tensors.
//
// Returns:
//   Number of input tensors
//
// Errors:
//   - If interpreter is not initialized
pub fn (i &Interpreter) get_input_tensor_count()! int {
	 if i.c_interpreter == 0 {
		return error('Interpreter is not initialized')
	}
	return int(C.TfLiteInterpreterGetInputTensorCount(i.c_interpreter))
}

// get_input_tensor returns the input tensor at the specified index.
//
// Parameters:
//   index - tensor index (0-based)
//
// Returns:
//   Tensor instance
//
// Errors:
//   - If interpreter is not initialized
//   - If tensor index is invalid
pub fn (i &Interpreter) get_input_tensor(index int) !Tensor {
	 if i.c_interpreter == 0 {
        return error('Interpreter is not initialized')
    }
	c_tensor := C.TfLiteInterpreterGetInputTensor(i.c_interpreter, i32(index))
	if c_tensor == 0 {
		return error('Failed to get input tensor at index ${index}')
	}
	return Tensor{c_tensor: c_tensor}
}

// resize_input_tensor resizes an input tensor to the specified dimensions.
// After calling this, you must call allocate_tensors() before running inference.
//
// Parameters:
//   index - tensor index (0-based)
//   dims - new dimensions for the tensor
//
// Errors:
//   - If interpreter is not initialized
//   - If dimensions array is empty
//   - If resize operation fails
//
// Example:
// ```v
// interpreter.resize_input_tensor(0, [1, 224, 224, 3])!
// interpreter.allocate_tensors()!
// ```
pub fn (i &Interpreter) resize_input_tensor(index int, dims []int) ! {
	if i.c_interpreter == 0 {
		return error('Interpreter is not initialized')
	}
	if dims.len == 0 {
		return error('Dimensions array is empty')
	}
	status := C.TfLiteInterpreterResizeInputTensor(i.c_interpreter, i32(index), dims.data,i32(dims.len))
	if status != .ok {
		return error('Failed to resize input tensor at index ${index}: status=${status}')
	}
}

// allocate_tensors allocates memory for all tensors.
// This must be called after resizing input tensors and before running inference.
//
// Errors:
//   - If interpreter is not initialized
//   - If allocation fails
//
// Example:
// ```v
// interpreter.resize_input_tensor(0, [1, 224, 224, 3])!
// interpreter.allocate_tensors()!
// ```
pub fn (i &Interpreter) allocate_tensors() ! {
		if i.c_interpreter == 0 {
		return error('Interpreter is not initialized')
	}
	status := C.TfLiteInterpreterAllocateTensors(i.c_interpreter)
	if status != .ok {
		return error('Failed to allocate tensors: status=${status}')
	}
}

// invoke runs the model inference.
// This executes the model and produces output tensors.
//
// Errors:
//   - If interpreter is not initialized
//   - If inference fails
//
// Example:
// ```v
// interpreter.set_input(0, input_data)!
// interpreter.invoke()!
// output_data := interpreter.get_output(0)!
// ```
pub fn (i &Interpreter) invoke() ! {
	if i.c_interpreter == 0 {
		return error('Interpreter is not initialized')
	}
	status := C.TfLiteInterpreterInvoke(i.c_interpreter)
	if status != .ok {
		return error('Failed to invoke interpreter: status=${status}')
	}
}

// get_output_tensor_count returns the number of output tensors.
//
// Returns:
//   Number of output tensors
//
// Errors:
//   - If interpreter is not initialized
pub fn (i &Interpreter) get_output_tensor_count() !int {
	if i.c_interpreter == 0 {
        return error('Interpreter is not initialized')
    }
	return int(C.TfLiteInterpreterGetOutputTensorCount(i.c_interpreter))
}

// get_output_tensor returns the output tensor at the specified index.
//
// Parameters:
//   index - tensor index (0-based)
//
// Returns:
//   Tensor instance
//
// Errors:
//   - If interpreter is not initialized
//   - If tensor index is invalid
pub fn (i &Interpreter) get_output_tensor(index int) !Tensor {
	if i.c_interpreter == 0 {
        return error('Interpreter is not initialized')
    }
	c_tensor := C.TfLiteInterpreterGetOutputTensor(i.c_interpreter, i32(index))
	if c_tensor == 0 {
		return error('Failed to get output tensor at index ${index}')
	}
	return Tensor{c_tensor: c_tensor}
}

// ============================================================================
// Tensor methods
// ============================================================================

// get_type returns the TensorFlow Lite type of the tensor.
//
// Returns:
//   TfLiteType enum value
//
// Example:
// ```v
// tensor_type := tensor.get_type()
// match tensor_type {
//   .float32 { println('Tensor is float32') }
//   .int32 { println('Tensor is int32') }
//   else { println('Unknown type') }
// }
// ```
pub fn (t &Tensor) get_type() TfLiteType {
	if t.c_tensor == 0 {
		return .no_type
	}
	return C.TfLiteTensorType(t.c_tensor)
}

// get_num_dims returns the number of dimensions of the tensor.
//
// Returns:
//   Number of dimensions (rank)
//
// Example:
// ```v
// num_dims := tensor.get_num_dims()
// println('Tensor rank: ${num_dims}')
// ```
pub fn (t &Tensor) get_num_dims() int {
	if t.c_tensor == 0 {
        return 0
    }
	return int(C.TfLiteTensorNumDims(t.c_tensor))
}

// get_dim returns the size of a specific dimension.
//
// Parameters:
//   dim_index - dimension index (0-based)
//
// Returns:
//   Size of the dimension
//
// Errors:
//   - If dimension index is out of range
//
// Example:
// ```v
// dim0 := tensor.get_dim(0)!
// dim1 := tensor.get_dim(1)!
// println('Tensor shape: ${dim0}x${dim1}')
// ```
pub fn (t &Tensor) get_dim(dim_index int) !int {
	if dim_index < 0 || dim_index >= t.get_num_dims() {
		return error('Dimension index ${dim_index} out of range [0, ${t.get_num_dims()})')
	}
	return int(C.TfLiteTensorDim(t.c_tensor, i32(dim_index)))
}

// get_dims returns all dimensions of the tensor as a slice.
//
// Returns:
//   Slice of dimension sizes
//
// Example:
// ```v
// dims := tensor.get_dims()!
// println('Tensor shape: ${dims}')  // e.g., [1, 224, 224, 3]
// ```
pub fn (t &Tensor) get_dims() ![]int {
	num_dims := t.get_num_dims()
	if num_dims == 0 {
		return []int{}
	}
	// Unoptimized version:
	// mut dims := []int{cap: num_dims}
	// for i in 0 .. num_dims {
	// 	dims << t.get_dim(i)!
	// }
	// Optimized version:
	mut dims := []int{len: num_dims}
		for i in 0 .. num_dims {
    		dims[i] = t.get_dim(i)!  // Directly assign to preallocated slice
		}
	return dims
}

// get_byte_size returns the size of the tensor data in bytes.
//
// Returns:
//   Size in bytes
//
// Example:
// ```v
// byte_size := tensor.get_byte_size()
// println('Tensor size: ${byte_size} bytes')
// ```
pub fn (t &Tensor) get_byte_size() usize {
	if t.c_tensor == 0 {
        return 0
    }
	return C.TfLiteTensorByteSize(t.c_tensor)
}

// get_name returns the name of the tensor.
//
// Returns:
//   Tensor name as a string
//
// Example:
// ```v
// name := tensor.get_name()
// println('Tensor name: ${name}')
// ```
pub fn (t &Tensor) get_name() string {
	if t.c_tensor == 0 {
        return('Tensor is not initialized')
    }
	c_name := C.TfLiteTensorName(t.c_tensor)
	if c_name == 0 {
		return ''
	}
	return unsafe { cstring_to_vstring(c_name) }
}

// copy_from_buffer copies data from a buffer into the tensor.
//
// Parameters:
//   data - pointer to the source buffer
//   size - number of bytes to copy
//
// Errors:
//   - If the copy operation fails
//
// Example:
// ```v
// mut data := [f32(1.0), 2.0, 3.0]!
// tensor.copy_from_buffer(data.data, data.len * sizeof(f32))!
// ```
fn (t &Tensor) copy_from_buffer(data voidptr, size usize) ! {
	status := C.TfLiteTensorCopyFromBuffer(t.c_tensor, data, size)
	if status != .ok {
		return error('Failed to copy data to tensor: status=${status}')
	}
}

// copy_from copies data from a V slice into the tensor.
//
// Parameters:
//   T - data type (e.g., f32, i32, u8, etc.)
//   data - slice of data to copy
//
// Errors:
//   - If the data type doesn't match the tensor type
//   - If the data size doesn't match the tensor size
//
// Example:
// ```v
// input_data := [1.0, 2.0, 3.0]!
// tensor.copy_from[f32](input_data)!
// ```
pub fn (t &Tensor) copy_from[T](data []T) ! {
    if !t.matches_type[T]() {
		return error('Type mismatch: tensor is ${t.get_type()}, but requested type does not match')
	}
    // 1. Check byte size
    element_size := int(sizeof(T))
    byte_size := data.len * element_size
    tensor_byte_size := int(t.get_byte_size())
    
    // 2. Check size match
    if byte_size != tensor_byte_size {
        return error('Size mismatch: input data is ${byte_size} bytes (${data.len} elements * ${element_size}), but tensor expects ${tensor_byte_size} bytes. Tensor dims: ${t.get_dims()!}')
    }
    
    // 3. Copy data
    t.copy_from_buffer(data.data, usize(byte_size))!
}

// copy_to_buffer copies data from the tensor to a specified buffer.
//
// Parameters:
//   buffer - pointer to the buffer to copy to
//   size - size of the buffer in bytes
//
// Errors:
//   - If the copy operation fails
//
// Example:
// ```v
// mut buffer := [0.0] * 100
// tensor.copy_to_buffer(buffer.data, buffer.len * sizeof(f64))!
// ```
fn (t &Tensor) copy_to_buffer(buffer voidptr, size usize) ! {
	status := C.TfLiteTensorCopyToBuffer(t.c_tensor, buffer, size)
	if status != .ok {
		return error('Failed to copy data from tensor: status=${status}')
	}
}

// copy_to copies data from the tensor into a V slice.
//
// Parameters:
//   T - data type (e.g., f32, i32, u8, etc.)
//   data - slice to copy data into (must be large enough)
//
// Errors:
//   - If the data type doesn't match the tensor type
//   - If the slice size doesn't match the tensor size
//
// Example:
// ```v
// mut output_data := [0.0, 0.0, 0.0]!
// tensor.copy_to[f32](mut output_data)!
// ```
pub fn (t &Tensor) copy_to[T](mut data []T) ! {
    if !t.matches_type[T]() {
		return error('Type mismatch: tensor is ${t.get_type()}, but requested type does not match')
	}
	// check buffer size for overflow
	element_size := int(sizeof(T))
	byte_size := data.len * element_size
	tensor_byte_size := int(t.get_byte_size())
	if byte_size != tensor_byte_size {
		return error('Size mismatch: output buffer is ${byte_size} bytes (${data.len} elements * ${element_size}), but tensor has ${tensor_byte_size} bytes. Tensor dims: ${
			t.get_dims()!}')
	}
	// byte_size := data.len * int(sizeof(T))
	t.copy_to_buffer(data.data, usize(byte_size))!
}

// ============================================================================
// Utility functions
// ============================================================================

// get_version returns the TensorFlow Lite version string.
//
// Returns:
//   Version string (e.g., "2.13.0") or empty string if unavailable
//
// Example:
// ```v
// version := tflite.get_version()
// println('TensorFlow Lite version: ${version}')
// ```
pub fn get_version() string {
	c_version := C.TfLiteVersion()
	if c_version == 0 {
		return ''
	}
	return unsafe { cstring_to_vstring(c_version) }
}

// ============================================================================
// InferenceResult
// ============================================================================

// InferenceResult stores output tensors from model inference.
// Uses TypedTensorData for zero-copy access to tensor data.
// WARNING: The data in TypedTensorData points to C-managed memory.
// Copy the data using copy_output[T]() if you need it after the ModelManager is freed.
pub struct InferenceResult {
pub mut:
	outputs []TypedTensorData  // Zero-copy access to output tensors
	dims    [][]int             // Dimensions of each output tensor
}


// ============================================================================
// ModelManager
// ============================================================================

pub struct ModelManager {
mut:
	model       Model
	interpreter Interpreter
	is_loaded   bool
	model_name  string
}

// new_model_manager creates a new ModelManager instance.
// The ModelManager manages the lifecycle of models and interpreters.
//
// Returns:
//   New ModelManager instance
//
// Example:
// ```v
// mut manager := new_model_manager()
// defer { manager.free() }
// manager.load_model("model.tflite", "test", 1, false)!
// ```
pub fn new_model_manager() ModelManager {
	return ModelManager{
		model: Model{c_model: unsafe { nil }}
		interpreter: Interpreter{
			c_interpreter: unsafe { nil }
			options: InterpreterOptions{
				c_options: unsafe { nil }
				xnnpack_delegate: unsafe { nil }
			}
		}
		is_loaded: false
		model_name: ''
	}
}

// load_model loads a TensorFlow Lite model and creates an interpreter.
//
// Parameters:
//   model_path - path to the .tflite model file
//   name - name for the model (used in error messages)
//   num_threads - number of threads for inference
//   xnn_options - whether to enable XNNPACK delegate 
//
// Errors:
//   - If a model is already loaded
//   - If model loading fails
//   - If interpreter creation fails
//   - If XNNPACK enable fails
//
// Example:
// ```v
// mut manager := new_model_manager()
// manager.load_model("model.tflite", "test", 4, true)!
// ```
pub fn (mut m ModelManager) load_model(model_path string, name string, num_threads int, xnn_options bool) ! {
	if m.is_loaded {
		return error('Model "${name}" is already loaded')
	}
	
	m.model_name = name
	
	m.model = create_model_from_file(model_path)!
	mut options := create_interpreter_options()
	
	options.set_num_threads(num_threads)
	if xnn_options {
		options.enable_xnnpack(num_threads) or {
			return error('Failed to enable XNNPACK delegate: ${err.msg()}')
		}
	}
	
	m.interpreter = create_interpreter_with_options(m.model, mut options)!
	m.interpreter.allocate_tensors()!
	
	m.is_loaded = true
}

// is_loaded checks if a model is currently loaded.
//
// Returns:
//   true if a model is loaded, false otherwise
//
// Example:
// ```v
// if manager.is_loaded() {
//     println('Model is loaded')
// }
// ```
pub fn (m &ModelManager) is_loaded() bool {
	return m.is_loaded
}

// get_model_name returns the name of the loaded model.
//
// Returns:
//   Model name string
//
// Example:
// ```v
// name := manager.get_model_name()
// println('Model name: ${name}')
// ```
pub fn (m &ModelManager) get_model_name() string {
	return m.model_name
}

// unload_model unloads the current model and releases resources.
// This is called automatically by free().
pub fn (mut m ModelManager) unload_model() {
	if m.is_loaded {
		m.interpreter.free()
		m.model.free()
		m.is_loaded = false
	}
}

// free releases all resources held by the ModelManager.
// This method should be called when the ModelManager is no longer needed.
// It is safe to call this method multiple times.
//
// Example:
// ```v
// mut manager := new_model_manager()
// manager.load_model("model.tflite", "test", 1, false)!
// manager.free()
// ```
pub fn (mut m ModelManager) free() {
	m.unload_model()
}

// set_input sets the data for an input tensor.
// This is a generic method that works with any supported type.
//
// Parameters:
//   T - data type (e.g., f32, i32, u8, etc.)
//   index - input tensor index (0-based)
//   data - slice of data to set
//
// Errors:
//   - If no model is loaded
//   - If tensor index is invalid
//   - If type mismatch
//   - If size mismatch
//
// Example:
// ```v
// input_data := [f32(1.0), 2.0, 3.0]!
// manager.set_input(0, input_data)!
// ```
pub fn (mut m ModelManager) set_input[T](index int, data []T) ! {
	if !m.is_loaded {
		return error('Model "${m.model_name}" is not loaded')
	}
	
	input_tensor := m.interpreter.get_input_tensor(index)!
	input_tensor.copy_from[T](data)!
}

// get_output returns a zero-copy slice pointing to the tensor's C-managed memory.
// WARNING: The slice becomes invalid if the ModelManager is freed or the model is unloaded.
// For safe usage, use copy_output[T]() instead.
pub fn (ir &InferenceResult) get_output[T](index int) ![]T {
    if index < 0 || index >= ir.outputs.len {
        return error('Output index ${index} out of range')
    }
    
    typed := ir.outputs[index]
    return typed.get_data[T]()!
}

// copy_output safely copies the output tensor data into a new V slice.
// This is safe to use even after the ModelManager is freed.
pub fn (ir &InferenceResult) copy_output[T](index int) ![]T {
    if index < 0 || index >= ir.outputs.len {
        return error('Output index ${index} out of range')
    }
    
    typed := ir.outputs[index]
    if !typed.matches_type[T]() {
        return error('Type mismatch: tensor is ${typed.tensor_type}, but requested type does not match')
    }
    
    mut data := []T{len: typed.size}
    
    // Use native V vmemcpy for efficient copying
    unsafe {
        vmemcpy(data.data, typed.data_ptr, isize(typed.size) * isize(sizeof(T)))
    }
    
    return data
}

// run_inference runs inference on the loaded model and returns output tensors.
//
// Returns:
//   InferenceResult containing zero-copy references to output tensors
//
// Errors:
//   - If no model is loaded
//   - If inference fails
//
// Example:
// ```v
// manager.load_model("model.tflite", "test", 1, false)!
// manager.set_input(0, input_data)!
// result := manager.run_inference()!
// output_data := result.copy_output[f32](0)!
// ```
// Note: int4 and int2 packed types are not fully supported.
// Use standard quantization (int8, uint8) for best compatibility.
pub fn (mut m ModelManager) run_inference() !InferenceResult {
	if !m.is_loaded {
		return error('Model "${m.model_name}" is not loaded')
	}
	
	m.interpreter.invoke()!
	
	output_count := m.interpreter.get_output_tensor_count()!
	mut result := InferenceResult{
		outputs: []TypedTensorData{cap: output_count}
		dims:    [][]int{cap: output_count}
	}
	
    for i in 0 .. output_count {
        output_tensor := m.interpreter.get_output_tensor(i)!
        dims := output_tensor.get_dims()!
        result.dims << dims
        
        tensor_type := output_tensor.get_type()
        byte_size := output_tensor.get_byte_size()
        
        // Special handling for int4 and int2 types
        total_elements := match tensor_type {
            .int4 {
                // 2 int4 values are packed into 1 byte
                int(byte_size) * 2
            }
            .int2 {
                // 4 int2 values are packed into 1 byte
                int(byte_size) * 4
            }
            .string, .resource, .variant {
                // Variable-size types: calculate from dimensions
                mut elements := 1
                for dim in dims {
                    elements *= dim
                }
                elements
            }
            else {
                // Fixed-size types (float32, int8, etc.)
                type_size := get_type_size(tensor_type)
                if type_size > 0 {
                    int(byte_size) / type_size
                } else {
                    // Unknown type - calculate from dimensions
                    mut elements := 1
                    for dim in dims {
                        elements *= dim
                    }
                    elements
                }
            }
        }
        
        result.outputs << TypedTensorData{
            tensor_type: tensor_type
            data_ptr: C.TfLiteTensorData(output_tensor.c_tensor)
            size: total_elements
        }
    }
    
    return result
}
