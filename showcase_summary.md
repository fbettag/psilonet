# Portfolio Showcase: psilonet (Psilocybin-Inspired LLM Architecture)

**Author:** Franz Bettag (fbettag)

**Target Role:** AI Research Engineer / Machine Learning Engineer (Targeting: OpenAI, Anthropic, DeepMind, Meta)

**Core Competencies:** LLM Architecture, Mechanistic Interpretability, On-Device Training (MLX), Systems Optimization.

---

## 🎯 Project Highlight: The Multi-Tap Skip Kernel in psilonet

**The Goal:**
Can we improve the reasoning and context integration of pre-trained LLMs without the massive cost of full retraining? Inspired by the "functional connectivity" increase observed in psychedelic neuroscience, I designed a novel "Skip-Layer" attention mechanism.

**The Solution:**
I engineered a **"Multi-Tap" kernel** that runs alongside a frozen pre-trained model (SmolLM2-135M). It allows layers to "time-travel"—attending directly to hidden states from 3, 4, or 5 layers prior. Crucially, the model **learns** which distance to use for each layer via a differentiable softmax gate.

### 🏆 Key Engineering Achievements

#### 1. MLX System Optimization (Apple Silicon)
*   **Problem:** Training complex multi-path kernels on a consumer GPU (M4 Pro) leads to rapid memory exhaustion.
*   **Solution:** Implemented **Manual Gradient Checkpointing** within the custom `PretrainedPsychedelicSmolLM` class. This required overriding the default `nn.value_and_grad` behavior to selectively recompute activations during the backward pass, enabling **3x larger batch sizes** and deeper kernels.
*   *Code:* `modules/psychedelic_smollm.py` (Custom Forward Pass with Checkpointing)

#### 2. Mechanistic Interpretability
*   **Problem:** "Performance" (lower loss) doesn't tell us *how* the model improved. Is it just memorizing?
*   **Solution:** Built a comprehensive analysis suite using **Centered Kernel Alignment (CKA)** and **SVCCA**.
*   **Insight:** Discovered a "Novelty Window" (Layers 9-27) where the model's representations diverge from the baseline to process complex dependencies, before re-converging for safe output generation.
*   *Artifact:* `experiments/cka_multitap.py` & `logs/cka/cka_per_head_side_by_side.png`

#### 3. Rigorous Hyperparameter Sweeps
*   **Problem:** Biological metaphors are noisy. How do we find the "Therapeutic Window"?
*   **Solution:** Conducted multi-seed ablation studies on the mixing coefficient ($\alpha$). Identified a statistically significant optimum at $\alpha=0.65$ (8.8% gain), proving the "inverted-U" curve predicted by neuroscience holds true for silicon neural networks.

---

## 💻 Tech Stack & Tools

*   **Languages:** Python 3.13, Shell Scripting
*   **ML Frameworks:** **MLX** (Deep expertise), PyTorch (Conceptual), HuggingFace Transformers
*   **Analysis:** NumPy, Matplotlib, CKA/SVCCA metrics
*   **DevOps:** Virtual environments, Git, CLI tooling

## 📚 Why This Matters

This project demonstrates independent research capability and creativity:
1.  **Original Ideation:** I hypothesized the "psychedelic connectivity" mechanism independently *before* validating it against neuroscience literature. This shows the ability to generate novel architectural ideas from first principles.
2.  **Systems Engineering:** Solving OOM errors on constrained hardware via manual gradient checkpointing in MLX.
3.  **Rigorous Validation:** Using interpretability methods (CKA/SVCCA) to ensure safety and understanding, moving beyond simple loss metrics.
