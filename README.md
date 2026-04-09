# Arcanon ML: CodeBERT Training Pipeline

This repository contains the training and fine-tuning pipeline for the Arcanon CodeBERT model, designed to enhance the Arcanon Scanner's static analysis with intelligent service-boundary inference.

## 🚀 RunPod Quickstart

Follow these steps to execute the training on a RunPod instance (recommended: **RTX 4090**).

### 1. Provision Your Pod
- **Image:** `pytorch/pytorch:latest`
- **Resource:** RTX 4090 (24GB VRAM)
- **Volume:** Mount a Network Volume to `/workspace` to ensure checkpoints persist.

### 2. Environment Setup
SSH into your pod and run:
```bash
# Clone the repository
git clone <your-repo-url>
cd arcanon-ml

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env and add your HF_TOKEN and WANDB_API_KEY
nano .env
```

### 3. Step-by-Step Execution

#### Step 1: Data Verification
Ensure your Hugging Face connection and streaming pipeline are working:
```bash
python data_check.py --n 5
```

#### Step 2: Phase 1 - Domain-Adaptive MLM
Teach the model modern cloud-native syntax (2024-2026). This typically takes 10-15 hours on a 4090.
```bash
python train_mlm.py \
  --push_to_hub \
  --hub_model_id your-username/arcanon-codebert-mlm \
  --report_to wandb \
  --fp16
```

#### Step 3: Phase 2 - Service-Context Classification (SBC)
Fine-tune for identifying service boundaries using your silver-standard labels.
```bash
python train_classifier.py \
  --train_file labels.jsonl \
  --push_to_hub \
  --hub_model_id your-username/arcanon-codebert-classifier \
  --report_to wandb \
  --fp16
```

#### Step 4: Export & Quantize
Convert the final model to a deployable INT8 ONNX file (~125MB).
```bash
python export_to_onnx.py \
  --model_dir ./classifier_checkpoints \
  --output_dir ./onnx_export
```

---

## 🛠 Configuration Guides

### Hugging Face Hub Integration
The training scripts include a `--push_to_hub` flag. To use it:
1.  **Get a Token:** Create a "Write" token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  **Add to .env:** Set `HF_TOKEN=your_token`.
3.  **Model ID:** Provide a `--hub_model_id` (e.g., `acme/my-model`).
4.  **Privacy:** By default, these scripts use your token to create private or public repos based on your HF account settings.

### Weights & Biases (WandB) Monitoring
Monitoring is essential for remote training on RunPod to avoid wasting your budget on failing runs.
1.  **Sign Up:** Create a free account at [wandb.ai](https://wandb.ai).
2.  **Get API Key:** Retrieve your key from [wandb.ai/authorize](https://wandb.ai/authorize).
3.  **Add to .env:** Set `WANDB_API_KEY=your_key`.
4.  **Run:** Pass `--report_to wandb` to the training scripts.
5.  **View:** Visit your WandB dashboard to see real-time loss curves and GPU health.

### Checkpoint Persistence
Checkpoints are saved to `./checkpoints` (Phase 1) and `./classifier_checkpoints` (Phase 2) by default. On RunPod, ensure these directories are within your mounted **Network Volume** so you can resume training if your instance is interrupted.

---

## 💻 Local Development (Apple Silicon / M3)

Your MacBook Pro M3 is excellent for smoke-testing the pipeline before deploying to RunPod. The scripts are configured to automatically detect and use the **MPS (Metal Performance Shaders)** device.

### 1. Local Setup
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your local secrets (ignored by git)
cp .env.example .env
nano .env  # Add your HF_TOKEN and WANDB_API_KEY
```

### 2. Run a "Smoke Test"
To verify the entire pipeline works without spending 15 hours, run a micro-training session (5 steps):

```bash
# Test Phase 1 (MLM)
python train_mlm.py --max_steps 5 --per_device_train_batch_size 1

# Test Phase 2 (Classification)
# Note: Requires a dummy 'labels.jsonl' file
python train_classifier.py --train_file labels.jsonl --num_train_epochs 1 --per_device_train_batch_size 1

# Test ONNX Export
python export_to_onnx.py --model_dir ./classifier_checkpoints
```

---

## 🔒 Security & Secrets

**CRITICAL:** Never commit your `.env` file, `HF_TOKEN`, or `WANDB_API_KEY` to GitHub.

### How to pass secrets to RunPod:
1.  **In the RunPod UI:** When creating a Pod or Serverless Endpoint, use the **"Environment Variables"** section.
2.  **Keys to set:**
    - `HF_TOKEN`: Your Hugging Face "Write" token.
    - `WANDB_API_KEY`: Your Weights & Biases API key.
3.  **Automated Handling:** The `run_all.sh` script and the Python training scripts will automatically detect these variables at runtime.

---

## 📄 File Overview
- `data_check.py`: Diagnostic for HF streaming pipeline.
- `train_mlm.py`: Masked Language Modeling (Phase 1).
- `train_classifier.py`: Classification Fine-tuning (Phase 2).
- `export_to_onnx.py`: ONNX export and INT8 quantization.
- `requirements.txt`: Python dependencies.
- `STRATEGY.md`: Deep dive into the training methodology.
