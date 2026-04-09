import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def check_data(num_samples=5):
    print(f"--- Starting Data Check (Streaming {num_samples} samples) ---")
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in .env file. Streaming might fail.")

    try:
        # Connect to the dataset
        dataset = load_dataset(
            "bigcode/the-stack-v2", 
            streaming=True, 
            split="train",
            token=token
        )

        )
        
        print("Successfully connected to The Stack v2.")
        
        for i, sample in enumerate(dataset.take(num_samples)):
            print(f"\n--- Sample {i+1} ---")
            # Using .get() with fallbacks for different Stack v2 schema versions
            print(f"Repository: {sample.get('repo_name', sample.get('repository_name', 'N/A'))}")
            print(f"Language:   {sample.get('lang', 'N/A')}")
            print(f"License:    {sample.get('license_type', 'N/A')}")
            print(f"File Path:  {sample.get('path', sample.get('directory', 'N/A'))}")
            
            content = sample.get('content', '')
            print(f"Content Snippet: {content[:100].replace('\\n', ' ')}...")
            print(f"Content Length:  {len(content)} characters")

    except Exception as e:
        print(f"\n[ERROR] Data check failed: {e}")
        print("Tip: Ensure your HF_TOKEN is valid and you have accepted the dataset terms on Hugging Face.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of samples to stream")
    args = parser.parse_args()
    check_data(args.n)
