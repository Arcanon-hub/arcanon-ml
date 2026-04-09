.PHONY: setup check-data test-mlm test-classifier test-export docker-build clean

# Setup local environment
setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Local setup complete. Please edit .env with your secrets."

# Verify data streaming from Hugging Face
check-data:
	python3 data_check.py --n 5

# Smoke test Phase 1 (MLM) - 5 steps only
test-mlm:
	python3 train_mlm.py --max_steps 5 --per_device_train_batch_size 1 --output_dir ./test_checkpoints

# Smoke test Phase 2 (Classification) - 1 epoch
# Note: Creates a dummy labels.jsonl if not present
test-classifier:
	@if [ ! -f labels.jsonl ]; then \
		echo '{"content": "import express; const app = express();", "label": 1}' > labels.jsonl; \
		echo '{"content": "function add(a, b) { return a + b; }", "label": 0}' >> labels.jsonl; \
	fi
	python3 train_classifier.py --train_file labels.jsonl --num_train_epochs 1 --per_device_train_batch_size 1 --model_name_or_path microsoft/codebert-base --output_dir ./test_classifier_checkpoints

# Test ONNX Export and Quantization
test-export:
	python3 export_to_onnx.py --model_dir ./test_classifier_checkpoints --output_dir ./test_onnx_export

# Build Docker container locally
docker-build:
	docker build -t arcanon-ml:latest .

# Cleanup temporary files and checkpoints
clean:
	rm -rf ./checkpoints ./classifier_checkpoints ./onnx_export ./test_checkpoints ./test_classifier_checkpoints ./test_onnx_export labels.jsonl
	find . -type d -name "__pycache__" -exec rm -rf {} +
