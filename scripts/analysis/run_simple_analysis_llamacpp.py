import os
from langchain_community.llms import LlamaCpp

# Replace with the path to your ggml compatible model
model_path = "./data/gguf_folder/llama-2-7b.Q4_0.gguf"

# Initialize LlamaCpp with GPU layers offloaded
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=1,       # Offload first layer to GPU for testing
    max_tokens=50,
    temperature=0.7
)

# Run a simple inference to verify it's working
prompt = "What is the capital of France?"
response = llm(prompt)

print("Prompt:", prompt)
print("Response:", response)
