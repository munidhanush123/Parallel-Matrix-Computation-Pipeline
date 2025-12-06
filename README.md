![C++ Version](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Taskflow](https://img.shields.io/badge/Taskflow-Library-green.svg)
![Build](https://img.shields.io/badge/Build-g++%20%7C%20CMake-orange.svg)
![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)

# ⚙️ Parallel Matrix Computation Pipeline Using Taskflow

## 📘 Introduction
This project demonstrates a modern parallel computing workflow using the Taskflow C++17 library. It implements tile-based (blocked) matrix multiplication and matrix addition inside a task dependency graph. The objective is to combine algorithmic optimization (tiling), concurrency (parallel tile execution), and structured scheduling (Taskflow DAG) to achieve fast, scalable, and correct matrix computations.

The system performs:
1. Task A — Random matrix generation  
2. Task B — Tile-based parallel matrix multiplication  
3. Task C — Tile-based parallel matrix addition  
4. Task D — Numerical verification, sequential baseline, and performance summary  

The method integrates blocked matrix multiplication, parallel C++ execution, and task-graph orchestration.

## 🧠 Method Overview
Naive matrix multiplication is slow due to poor data locality. Blocked (tiled) GEMM restructures the algorithm to operate on small submatrices that fit into CPU cache. Each tile computes:
C(I, J) += A(I, K) × B(K, J)

Tiles are independent, allowing parallel execution. Taskflow expresses this as a Task Dependency Graph where parallelizable tile tasks are executed using subflows.

## 🛠️ Installation Instructions

Install dependencies:
```bash
sudo apt update
sudo apt install g++ cmake git graphviz
```
Clone this repository:
```bash
git clone https://github.com/KundhanMiriyala/parallel-taskflow-case-study.git
cd parallel-taskflow-pipeline
```

Download Taskflow:
```bash
mkdir -p 
cd deps
git clone https://github.com/taskflow/taskflow.git deps/taskflow
```

Build (g++):
```bash
g++ -std=c++17 main.cpp -I deps/taskflow -pthread -O3 -o taskflow_matrix
```

Or using CMake:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
## 🚀 Usage
Default:
```bash
./taskflow_matrix
```

Options: 
```txt
-n <size>   Matrix dimension 
-t <tile>   Tile size 
-s <seed>   Random seed 
--no-dot    Disable DAG output
```

Example:
```bash
./taskflow_matrix -n 1024 -t 128 -s 42
```

## 🌍 Real-World Applications
- HPC dense linear algebra  
- Scientific computing  
- CPU tensor kernels in ML frameworks  
- DAG execution engines  
- Performance-engineered C++ workloads  

## ✨ Features
- Taskflow DAG (A → {B, C} → D)
- Tile-based parallel GEMM
- Tile-based parallel matrix addition
- Subflow-based parallelism
- Sequential baseline verification
- Timing and speedup computation
- DOT graph export
- Fully configurable tile & matrix sizes

## 🧮 Algorithm Explanation
- Blocked GEMM: <br>
C(I,J) = ΣK A(I,K) × B(K,J) <br>
Reduces cache misses; improves throughput; used in BLAS/MKL. <br>


- Parallelism: <br>
One task per tile. Taskflow schedules tile tasks across CPU cores. <br>


- Verification: <br>
Sequential GEMM used as a reference.
Floating-point tolerance 1e-6.

## 📊Example Output Summary
```txt
Matrix N=512, tile=64 
A: Generated matrices 
C: Blocked addition complete 
B: Blocked multiplication complete 

Speedup, timing breakdown, and verification results printed. 

DOT graph saved to taskflow_graph.dot 
```


## 📚 References

1. **Taskflow Official Documentation**  
   https://taskflow.github.io/taskflow/

2. **Taskflow GitHub Repository (Examples & API Reference)**  
   https://github.com/taskflow/taskflow

3. **Blocked / Tiled Matrix Multiplication (GEMM) — Classical Literature**  
   - Kazushige Goto & Robert A. van de Geijn, *Anatomy of High-Performance Matrix Multiplication*  
   - BLIS Framework Documentation: https://github.com/flame/blis  
   - OpenBLAS Reference: https://github.com/xianyi/OpenBLAS

4. **C++17 Standard Library (Concurrency, Atomics, Memory Model)**  
   https://en.cppreference.com/w/
## 📄 License
For academic and educational use.
