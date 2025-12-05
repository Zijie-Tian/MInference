# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from minference import MInference


def run_target_length(m: int, llm, tokenizer, sampling_params, attn_type: str):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)

    s = 0
    T = 10
    for i in range(T + 1):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        if i:  # skip warmup
            s += time.time() - start
    print(attn_type, m, s / T)
    return s / T


def run_benchmark(model_name: str, attn_type: str):
    TARGET_LENS = [l * 1000 for l in [4, 8]]
    ATTN_TYPES = ["flash_attn", "minference"]
    ATTN_TYPES2NAME = {
        "flash_attn": "FlashAttention-2",
        "minference": "MInference",
    }
    latency = defaultdict(list)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

    for attn_type in ATTN_TYPES:
        max_len = TARGET_LENS[-1] + 10_000
        llm = LLM(
            model_name,
            enforce_eager=True,
            max_model_len=max_len,
            enable_chunked_prefill=False,
        )
        if attn_type == "minference":
            minference_patch = MInference("vllm_minference", model_name)
            llm = minference_patch(llm)

        for l in TARGET_LENS:
            t = run_target_length(l, llm, tokenizer, sampling_params, attn_type)
            latency[ATTN_TYPES2NAME[attn_type]].append([l, f"{t:.5f}"])
            print(attn_type, t, l)
            torch.cuda.empty_cache()

        del llm
        torch.cuda.empty_cache()

    res = [[""] + [ATTN_TYPES2NAME[attn_type] for attn_type in ATTN_TYPES]]
    for idx in range(len(TARGET_LENS)):
        l = TARGET_LENS[idx]
        res.append(
            [f"{l//1000}K"]
            + [latency[ATTN_TYPES2NAME[attn_type]][idx][-1] for attn_type in ATTN_TYPES]
        )
    print("\n".join(["\t".join(ii) for ii in res]))
    with open("res_vllm.csv", "w") as f:
        f.write("\n".join(["\t".join(ii) for ii in res]))
    return res


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    args.add_argument(
        "--attn_type",
        type=str,
        choices=["flash_attn", "minference"],
    )
    args.add_argument("--context_window", type=int, default=100_000)
    args.add_argument("--run_benchmark", action="store_true")
    args = args.parse_args()

    model_name = args.model_name

    if args.run_benchmark:
        run_benchmark(model_name, args.attn_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

        llm = LLM(
            model_name,
            enforce_eager=True,
            max_model_len=args.context_window + 10_000,
            enable_chunked_prefill=False,
        )

        # Patch MInference Module
        if args.attn_type == "minference":
            minference_patch = MInference("vllm_minference", model_name)
            llm = minference_patch(llm)

        run_target_length(args.context_window, llm, tokenizer, sampling_params, args.attn_type)
