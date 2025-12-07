# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Example of using MInference with vLLM.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 python examples/run_vllm.py

Note: VLLM_USE_V1=0 is required because MInference only supports vLLM V0 engine.
"""

import os

# Force vLLM to use V0 engine (must be set before importing vllm)
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams

from minference import MInference


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=20,
    )

    # Use local model path if available, otherwise use HuggingFace model
    model_name = "gradientai/Llama-3-8B-Instruct-262k"
    local_model_path = "/home/zijie/models/Llama-3-8B-Instruct-262k"
    if os.path.exists(local_model_path):
        model_name = local_model_path

    print(f"Loading model: {model_name}")

    llm = LLM(
        model_name,
        max_num_seqs=1,
        enforce_eager=True,
        max_model_len=32000,  # Reduced for faster testing
        tensor_parallel_size=1,
    )

    # Patch MInference Module
    print("Applying MInference patch...")
    minference_patch = MInference("vllm", model_name)
    llm = minference_patch(llm)

    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 40)


if __name__ == "__main__":
    main()
