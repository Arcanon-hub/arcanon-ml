# Arcanon ML: CodeBERT Training Strategy

This document tracks the design and execution of a specialized CodeBERT model to enrich the Arcanon Scanner's static analysis.

## 1. Goal: From Static Analysis to Intelligent Inference
The current Arcanon Scanner uses tree-sitter to extract service connections. However, it struggles with:
- **Obfuscated/Dynamic Targets:** When a URL is constructed at runtime (e.g., `process.env.HOST + "/v1"`).
- **Ambiguous Boundaries:** Differentiating between a generic library and a core service entry point.
- **Protocol Prediction:** Guessing whether an open socket is using gRPC, REST, or a custom MQ protocol.

**Our solution:** Fine-tune `microsoft/codebert-base` on modern GitHub data (2024-2026) to act as a "Confidence Layer" for the scanner.

---

## 2. Data Pipeline: "The Stack v2" Streaming & Filtering

We will use the Hugging Face `datasets` library in **streaming mode** to avoid downloading the multi-terabyte dataset.

### 2.1 Language & Metadata Filters
- **Scope:** Filter for `lang` in `[rust, go, python, java, typescript, javascript, ruby, csharp]`.
- **License Guard:** Only process files with metadata matching `license_type == "permissive"`.
- **Temporal Filter:** Focus on repositories with a `last_commit_date` after **January 1, 2024**. This is critical because CodeBERT (2020) has never seen modern frameworks like FastAPI 0.100+, Spring Boot 3.x, or newer Rust crates.

### 2.2 Content-Based Sampling
We don't need every `index.ts` file. We will use a "Heuristic Sampler" to prioritize files that look like service glue:
- **Heuristic 1 (Imports):** Prioritize files importing `express`, `axum`, `tokio`, `requests`, `boto3`, `kafka-python`, etc.
- **Heuristic 2 (Annotations):** Prioritize files with decorators like `#[get("/")]`, `@app.get`, `@Controller`.
- **Heuristic 3 (Structure):** Prioritize files that define a "Client" or "Service" class.

---

## 3. Training Objectives: Domain-Adaptive Fine-tuning

We will use a two-stage training process on our RunPod instance.

### Phase 1: Domain-Adaptive MLM (Masked Language Modeling)
- **What:** Randomly mask 15% of the tokens in our "Modern GitHub" corpus and have the model predict them.
- **Why:** This forces the model to learn the specific syntax and context of 2026-era cloud-native code.
- **Scale:** 10-15 hours of training.

### Phase 2: Service-Context Classification (SBC)
- **What:** Add a classification head to the `[CLS]` token of CodeBERT.
- **Labeling (Silver-Standard):** We will use the *existing* Arcanon Rust Scanner to "label" our training data.
    - If Arcanon detects a high-confidence service boundary (e.g., a clear FastAPI route), we label it `1`.
    - If Arcanon detects a utility function with no external connections, we label it `0`.
- **Goal:** The model learns to "guess" boundaries in cases where Arcanon's static rules are too strict or too loose.

---

## 4. RunPod Orchestration ($25 Budget)

We have roughly **33 hours** on an RTX 4090. Every hour must be productive.

### 4.1 Cost-Efficient Setup
1. **Container Image:** Use `pytorch/pytorch:latest` with pre-installed `transformers` and `accelerate`.
2. **Persistence:** Use a **RunPod Network Volume** for the `/workspace/checkpoints` folder. If the pod is interrupted, we resume from the last 500 steps.
3. **Data Throttling:** The bottleneck is usually the CPU-based data streaming. We will use `IterableDataset` with multiple worker threads to keep the GPU utilization at >90%.

### 4.2 Hyperparameters for RTX 4090
- **FP16:** Mandatory. Reduces VRAM usage and doubles speed.
- **Batch Size:** 16 per step.
- **Gradient Accumulation:** 4 steps. This gives us a "Virtual Batch Size" of 64, which is necessary for stable BERT convergence.
- **Learning Rate:** 5e-5 (standard for fine-tuning).

---

## 5. Integration: The "Bridge" to Rust

The final model must be small and fast.
- **Export to ONNX:** We will export the final PyTorch weights to the Open Neural Network Exchange format.
- **INT8 Quantization:** We will use `onnxruntime` tools to quantize the model. This will bring the size down from ~490MB to **~125MB**, making it small enough to ship inside a CLI tool.
- **Inference in Rust:** Use the `ort` crate. The scanner will call the model *only* when it encounters "Low Confidence" detection paths to minimize CPU overhead.

---

## 6. Immediate Action Items
1. [ ] **`requirements.txt`**: Define the Python stack (PyTorch, Transformers, Datasets, Accelerate).
2. [ ] **`data_check.py`**: A script to stream 1,000 files from "The Stack v2" and print their metadata to verify our filters.
3. [ ] **`train_mlm.py`**: The core training script with checkpointing logic.
