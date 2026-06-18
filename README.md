# Behavior-Guided Multimodal Policy Search

This repository contains the official implementation for the NeurIPS paper:

> **Behavior-Guided Multimodal Policy Search: Leveraging Vision-Language Models for Reinforcement Learning**

## Technical Report

The full technical report for this project is available at the following link:

- [Technical Report (Google Drive)](https://drive.google.com/drive/folders/1wo-JWb6zx5pGAp9Xdd5sQJ8Iu1JiX60X)

## Methods

This codebase implements three methods:

- **ProPS** (`props`): Uses LLMs to directly optimize a reinforcement learning policy via numerical parameter updates.
- **ProPS+** (`propsp`): Extends ProPS by incorporating domain-specific descriptions as semantic context in the prompt to guide optimization.
- **BMPS** (`bmps`): Behavior Landscape-guided Multi-modal Policy Search that integrates vision-language model feedback with policy optimization for visual RL tasks.

See [`llm-rl-main/readme.md`](llm-rl-main/readme.md) for installation and usage details.
