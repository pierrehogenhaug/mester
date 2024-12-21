# import llama_cpp
# print(llama_cpp.llama_print_system_info())

from llama_cpp import Llama

# Initialize the Llama model with GPU support
llm = Llama(
    model_path="./data/gguf_folder/llama-2-7b.Q4_0.gguf",  # Update this path to your actual model location
    n_gpu_layers=35  # Adjust based on GPU memory; 32GB GPU should handle ~35 layers
)

# Perform inference
output = llm(
    "Q: Name the planets in the solar system? A: ",
    max_tokens=32,
    stop=["Q:", "\n"],
    echo=True
)

# Print the output
print(output)