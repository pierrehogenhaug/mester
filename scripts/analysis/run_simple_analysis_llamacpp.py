import llama_cpp

# Print system information to verify GPU detection
print("System Information:")
print(llama_cpp.llama_print_system_info())

from llama_cpp import Llama

def main():
    # Initialize the Llama model with GPU support
    print("\nInitializing Llama model...")
    llm = Llama(
        model_path="./data/gguf_folder/llama-2-7b.Q4_0.gguf",  # Ensure this path is correct
        n_gpu_layers=35  # Adjust based on GPU memory; 32GB GPU should handle ~35 layers
    )
    print("Llama model initialized successfully.")

    # Perform inference
    print("\nPerforming inference...")
    output = llm(
        "Q: Name the planets in the solar system?",
        max_tokens=32,
        stop=["Q:", "\n"],
        echo=True
    )

    # Print the output
    print("\nInference Output:")
    print(output)

if __name__ == "__main__":
    main()