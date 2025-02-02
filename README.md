
# Synapse 💡

**Lighting the way.**  
Combines the safety of Rust with the simplicity of Python and the efficiency of C.  
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://chat.qwenlm.ai/c/LICENSE)  
[![Status: Active Development](https://img.shields.io/badge/status-Active%20Development-orange)](https://github.com/synapse-lang/synapse)

## Key Features

-   ✅ **Automatic Ownership** : Memory safety without explicit annotations.
-   🔄 **Automatic Move** : Values are transferred without unnecessary copies.
-   🔒 **Safe Borrowing** : The compiler ensures that borrows don't outlive the original data.
-   ⚡ **Zero "use-after-move"** : Compile-time detection of invalid accesses.
-   🧩 **Multiple Dispatch** : Function specialization based on argument types.
-   🐍 **Intuitive Syntax** : Familiar to Python/Julia users.
-   🔄 **Interoperability** : Direct integration with Python (PyTorch, NumPy).
-   🚀 **Native Performance** : AOT/MLIR compilation.
-   📦 **Package Manager** : Inspired by Cargo (Rust).
-   🚀 **Cross-Platform Compilation** : Native support for x86, PowerPC, ARM, IBM Z (s390x), WebAssembly, RISC-V, and GPUs.
-   🧠 **Optimized for AI** : Excellent integration with CUDA and PTX for NVIDIA GPUs.

## Installation (dev)

```bash
# Requires Rust and LLVM 15+
git clone https://github.com/synapse-lang/synapse
cd synapse
cargo build --release
```
## Benchmarks (Expected)

|Operation | Synapse| Pyhon|
|--|--|--|
|Load CSV (10 GB)| 1.2s| 12.4s |
|Filter + Map| 0.8s| 3.1s |
|Train model| 4.5s|15.3s |

## Advantages vs Rust

|**Case**|**Rust**|**Synapse**|
|----------------|-------------------------------|-----------------------------|
| Borrowing |`fn f(a: &T)`| Inferred by the compiler
|Common Errors|explicit `borrow-checker`| Clear post-compilation messages

## Compilation

Synapse allows you to compile your code for multiple architectures and devices:
```python
synapse compile --optimize --target [cpu|x86|arm|powerpc|s390x|riscv|wasm|gpu] my_program.sm
```

# Syntax in 60 Seconds

## Variables and Functions

```python
let x: Int = 5;           // Explicit type
let y = 10;               // Type inference
func sum(a: Int, b: Int) -> Int {
    return a + b;
}
```


## Structures and Methods
```python
type Vector2D {
    x: Float;
    y: Float;
}
impl Vector2D {
    fn magnitude(&self) -> Float {
        sqrt(self.x^2 + self.y^2);
    }
}
let v = Vector2D { x: 3.0, y: 4.0 };
print(v.magnitude());      // 5.0
```

## Enums and Aliases
```python
enum Result {
    Success(value: Int);
    Error(message: String);
}
alias ID = String;
let id: ID = "user_123";
```

## Lazy Evaluation
```python
// Fibonacci generator (infinite, on-demand)
lazy fib = {
    let (a, b) = (0, 1);
    loop {
        yield a;
        (a, b) = (b, a + b);
    }
}

// Evaluates only the first 5 terms
let first5 = fib.take(5).collect();  // [0, 1, 1, 2, 3]
```

## Multiple Dispatch
```python
func process(a: Int, b: Int) -> Int { a + b; }
func process(a: String, b: String) -> String { a ++ b; }
print(process(2, 3));          // 5
print(process("Hello", "Synapse")); // "HelloSynapse"
```

## Key Optimizations

-   **Copy-On-Write (COW)** : Avoids data copies until necessary.
-   **Operation Fusion** : Combines `map`, `filter`, etc., into a single loop.
-   **Memoization** : Caches results of pure functions (`@pure`).

```python
let result = data  // Example of automatic fusion
    .map(filter)
    .filter(is_valid)
    .take(10_000);  // Single pass, no intermediate copies
```

## Implicit Ownership
### How It Works

1.  **Assignment** : Each value has a unique owner.
2.  **Move** : When passing values to functions or assigning variables, ownership is transferred.

```python
let data = [1, 2, 3];
let result = process(data);  
print(data); // ❌ Error: Moved (data is no longer valid)
```

### Safety Without Effort
```python
type Sensor { value: Float }

func read_sensor(s: Sensor) -> Float {
    s.value * 5  // Ownership moves here
}

let sensor1 = Sensor { value: 25.0 };
let sensor1_clone = sensor1.clone();      // Explicit copy
let measurement = read_sensor(sensor1);   // Original moved (✅)
let measurement2 = read_sensor(sensor1_clone);   // Use of the copy (✅)
```
### **Why No Automatic Copies?**

-   **Avoid performance surprises** : Copying large matrices without consent is costly in data science.
-   **Programmer control** : You decide what, when, and how to clone.

## Ownership-Based Optimizations

-   **Memory reuse** : The compiler automatically destroys values when they go out of scope.
-   **Elided copies** : Uses internal references where safe.
-   **Concurrency without locks** : Actors receive unique ownership of messages.

## Interoperability with Python
```python
from python.pandas import DataFrame;
let df = DataFrame::from_csv("data.csv");
print(df.head());
let processed = process(df);  // Ownership managed even with Python objects
```
```python
from python.numpy import ndarray;

let matrix = [[1.0, 2.0], [3.0, 4.0]];
let np_array = matrix.to_numpy();  # Read-only array in Python
print(np_array.flags.writeable);   # False
```
## Concurrency (Actors)
```python
actor Counter {
    init() -> Int { 0 }
    handle("increment", count: Int) -> (Int, String) {
        (count + 1, "OK");
    }
}
let counter = Counter::spawn();
counter.send("increment", 5);  // (6, "OK")
```
## Type System

|Construt |Example                         |Description  |
|----------------|-------------------------------|-----------------------------|
|type       |type Sensor { value: Float }   |Data structures.
|enum       |enum Option { yes; No }         |Enumerations with/without data.
|alias      |alias Matrix = Array[Float, 2] |Type synonyms.

### GPU Optimization

Synapse is designed to fully leverage modern GPUs, especially in AI applications. Highlights include:

#### PTX (Parallel Thread Execution) Support

-   Direct generation of optimized PTX code for NVIDIA GPUs.
-   Simplified abstraction for working with CUDA.
-   Basic GPU kernel example:
```python
@gpu_kernel
func process_data(data: Array[Float, 1]) {
    let idx = threadIdx.x + blockIdx.x * blockDim.x;
    if idx < data.len() {
        data[idx] = tanh(data[idx]);
    }
}
```
#### AI Advantages

-   **Massive parallelism** : Leverages thousands of cores simultaneously.
-   **Unified memory** : Transparent management between RAM and VRAM.
-   **Native autodiff** : Optimized automatic differentiation for GPUs.
-   **PyTorch interoperability** : Direct conversion between tensors.
```python
from python.torch import Tensor;

let model = Sequential(
    Dense(784, 256),
    ReLU(),
    Dense(256, 10)
);

func train(data: Tensor, labels: Tensor) -> Float {
    let predictions = model(data.gpu());
    let loss = cross_entropy(predictions, labels.gpu());
    backward(loss);
    return loss.cpu().item();
}
```

## Roadmap 2025

-   MVP Compiler (Q2 2025): Support for structs, functions, and Python FFI.
-   Safe Concurrency (Q3 2025): Actors and asynchronous channels.
-   Ecosystem (Q4 2025): Package manager and standard library.

## Contribute

Open an issue or submit a PR.

## License

MIT © 2025 Synapse Project.

