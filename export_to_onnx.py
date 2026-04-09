import os
import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_hub(folder_path, repo_id):
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not found in environment. Skipping upload.")
        return
    
    logger.info(f"Uploading {folder_path} to Hugging Face Hub: {repo_id}")
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        logger.info("Upload complete!")
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face Hub: {e}")

def export_onnx(model_path, output_path):
    logger.info(f"Loading model from {model_path} for ONNX export...")
    
    # Load to CPU for export to ensure device-agnostic graph
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    model.eval()

    # 2. Prepare dummy input for ONNX tracing
    # CodeBERT/RoBERTa expects input_ids and attention_mask
    dummy_input = tokenizer("This is a dummy service check", return_tensors="pt")
    
    onnx_file = os.path.join(output_path, "model.onnx")
    
    # 3. Export to ONNX
    logger.info("Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    logger.info(f"FP32 ONNX saved to {onnx_file}")
    return onnx_file

def quantize_onnx(onnx_file, output_path):
    # 4. Apply INT8 Quantization
    quantized_file = os.path.join(output_path, "model_quantized.onnx")
    logger.info("Starting INT8 Dynamic Quantization...")
    
    quantize_dynamic(
        model_input=onnx_file,
        model_output=quantized_file,
        weight_type=QuantType.QInt8
    )
    
    orig_size = os.path.getsize(onnx_file) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_file) / (1024 * 1024)
    
    logger.info(f"Quantization Complete!")
    logger.info(f"Original Size:  {orig_size:.2f} MB")
    logger.info(f"Quantized Size: {quant_size:.2f} MB")
    logger.info(f"Final Model Saved to: {quantized_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned CodeBERT to Quantized ONNX")
    parser.add_argument("--model_dir", type=str, default="./classifier_checkpoints", help="Path to your fine-tuned classifier")
    parser.add_argument("--output_dir", type=str, default="./onnx_export", help="Where to save the ONNX files")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the ONNX files to HF Hub")
    parser.add_argument("--hub_model_id", type=str, help="HF repository name (e.g., 'user/repo')")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    onnx_fp32 = export_onnx(args.model_dir, args.output_dir)
    quantize_onnx(onnx_fp32, args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        upload_to_hub(args.output_dir, args.hub_model_id)
