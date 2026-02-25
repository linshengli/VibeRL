# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a knowledge/documentation repository containing a comprehensive Chinese-language guide for reproducing a **Vibe Coding Agent** with **RL post-training**. It is based on gaocegege's blog post and covers the full pipeline: Multi-Agent architecture, SFT (Supervised Fine-Tuning), and GRPO/NGRPO reinforcement learning.

The project consists of a single guide file: `从零复现_Vibe_Coding_Agent到RL后训练_完整指南.md`

## Domain Context

The guide describes a 5-phase implementation for building an LLM-based stock analysis agent as a proof-of-concept:

- **Phase 0**: Environment setup (GPU, CUDA, PyTorch, verl, vLLM)
- **Phase 1**: Tool-use Agent construction using the ReAct (Reasoning + Acting) pattern
- **Phase 2**: SFT data generation and training
- **Phase 3**: GRPO/NGRPO reinforcement learning training
- **Phase 4**: Multi-Agent architecture extension (optional)

Key technologies referenced: verl (RL framework), vLLM (inference), Qwen2.5-7B-Instruct, DeepSeek models, GRPO, NGRPO, DPO, FSDP, PyTorch, Hugging Face ecosystem.

## Working with This Repository

- The guide is written in Chinese (Simplified). Respond in the same language as the user's query.
- When editing the guide, preserve its existing structure (numbered phases, markdown tables, code blocks).
- Code examples in the guide target Python 3.10+ with PyTorch 2.1+ and CUDA 12.1+.
