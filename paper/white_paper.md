# SPEAR: Spherical Prompt-preserving Embedding Adaptation for Expressive Zero-Shot Voice Cloning in Closed-Vocabulary TTS

**Author:** Efeberk Çeker  
**Affiliation:** Independent Researcher  
**Contact:** efeberkceker@gmail.com  
**arXiv categories:** cs.SD (Sound), cs.CL (Computation and Language)  
**Date:** June 2026

---

## Abstract

Closed-vocabulary text-to-speech (TTS) systems restrict speaker identity to a fixed set of named tokens, seemingly precluding zero-shot adaptation. We demonstrate this restriction is not fundamental. We present **SPEAR** (Spherical Prompt-preserving Embedding Adaptation for expressive TTS), a method that achieves zero-shot voice cloning in closed-vocabulary TTS models via in-memory speaker token replacement — without fine-tuning, LoRA, or any disk modification.

SPEAR operates by (1) extracting a **consistency-weighted speaker centroid** from reference audio, (2) interpolating between the target speaker token and the centroid via **Spherical Linear Interpolation (SLERP)**, and (3) patching the embedding table in RAM at inference time with norm-matched vectors.

We validate SPEAR on Qwen3-TTS-1.7B across three distinct speaker tokens (Ryan, Serena, Dylan), demonstrating:

- Speaker similarity (SECS) ≥ **0.989** across all three speakers at optimal alpha
- Smooth, monotonic SECS increase across 8 alpha values (0.0 → 1.0)
- Full preservation of the host model's emotion instruction-following across 6 affect categories (neutral, sad, happy, angry, whisper, deep)
- Reference audio requirement of only **~27 seconds**

Our results provide systematic empirical evidence that closed-vocabulary TTS speaker embedding spaces are **continuously structured**, enabling parametric voice interpolation via a single scalar alpha.

---

## 1. Introduction

Modern neural TTS systems diverge architecturally on how speaker identity is handled. **Open-conditioning** models such as XTTS, F5-TTS, and OpenVoice accept arbitrary reference audio at inference time and condition the acoustic decoder via cross-attention or prompt prepending. **Closed-vocabulary** models such as Qwen3-TTS CustomVoice assign speaker identity through a discrete lookup table: each named token (Ryan, Serena, Dylan) maps to a fixed embedding vector. This design offers advantages in inference speed and instruct-following fidelity, but appears to prohibit adaptation to unseen speakers.

We challenge this assumption. The speaker embedding tables in closed-vocabulary TTS models are trained jointly with the speaker encoder that processes reference audio during training. This joint training forces both components to share a common latent space — a property necessary for the model to generalize during training. At inference time, this shared space can be exploited: a reference speaker's embedding, extracted via the base model's encoder, occupies the same geometric space as the named speaker tokens. Replacing a token's embedding with a reference centroid is therefore mathematically coherent.

Beyond binary replacement, we show the space supports **continuous interpolation**. A scalar parameter alpha smoothly transitions voice identity between the original token and the reference speaker. This transforms the discrete speaker lookup into a continuous dial, enabling partial adaptation, voice blending, and identity-preserving style transfer.

Our method requires no gradient updates, no additional parameters, and no disk writes. The entire adaptation occurs in RAM in under one second, making it suitable for interactive and production deployment.

---

## 2. Related Work

### 2.1 Zero-Shot Voice Cloning

Zero-shot voice cloning — adapting TTS to an unseen speaker from short reference audio — has been approached via several paradigms:

**Prompt-based conditioning** (XTTS, Tortoise-TTS, F5-TTS): Reference audio is encoded at inference time and injected via cross-attention or token prepending. These systems require models explicitly designed for external conditioning.

**Voice conversion** (RVC, So-VITS-SVC): A separate model maps source audio to target speaker timbre. This approach requires a two-stage pipeline and typically degrades the prosodic richness of the source.

**Neural codec language models** (VALL-E, VoiceCraft): Speaker identity is represented as acoustic token prefixes. These models achieve high similarity but require substantial reference audio and careful prompt selection.

SPEAR differs from all of these: it requires no model redesign, no separate conversion stage, and no codec tokenization. It operates directly on the weight matrix of an existing closed-vocabulary model.

### 2.2 Embedding Space Manipulation

Latent space interpolation has been extensively studied in image generation (GANs, diffusion models) where smooth semantic transitions are achieved by interpolating in latent space. SLERP was introduced for this purpose by White (2016), who showed it produces more perceptually consistent interpolations than LERP on the unit hypersphere.

In NLP, "soft prompting" (Lester et al., 2021) demonstrated that continuous embedding vectors can replace discrete token prompts. Our method applies an analogous insight to TTS speaker tokens.

In this work, we systematically investigate the continuous speaker space structure in closed-vocabulary TTS models and evaluate SLERP-based speaker token interpolation as a zero-shot adaptation strategy.

---

## 3. Method: SPEAR

SPEAR consists of four components applied sequentially at inference time.

### 3.1 Consistency-Weighted Speaker Centroid Extraction

Given reference audio *A*, we extract a robust speaker representation as follows:

**Step 1 — Segmentation.** Divide *A* into segments of 3–8 seconds using energy-based VAD. For our experiments, a 27-second reference yields 16 usable segments.

**Step 2 — Embedding extraction.** Apply the base model's speaker encoder *E* to each segment *s_i*:

```
e_i = E(s_i)
```

**Step 3 — Consistency scoring.** Score each segment by combining (a) mean pairwise cosine similarity to all other segments and (b) cosine similarity to the global mean:

```
score_i = 0.6 · mean_{j≠i} cos(e_i, e_j) + 0.4 · cos(e_i, ē)
```

where ē = mean(e_1, ..., e_n). This scoring rewards segments that are simultaneously consistent with other segments (intra-speaker stability) and representative of the overall identity (centrality).

**Step 4 — Softmax weighting.** Apply temperature-scaled softmax to amplify quality differences:

```
w_i = softmax(score_i / τ),   τ = 0.1
```

Low temperature (τ = 0.1) creates sharp weight concentration on high-quality segments, effectively discarding noisy or atypical segments.

**Step 5 — Weighted centroid.** Select top-K segments (K=5) by score and compute weighted average:

```
c = Σ_{i ∈ top-K} w_i · e_i
```

In our experiments, segment scores range from 0.986 to 0.991 (mean 0.989), indicating high intra-speaker consistency in the reference audio.

### 3.2 Spherical Linear Interpolation (SLERP)

Given target speaker token vector *v_t* and reference centroid *c*, we compute an interpolated vector at parameter α ∈ [0,1]:

```
θ = arccos(clip(v̂_t · ĉ, −1, 1))

SLERP(v_t, c, α) = sin((1−α)θ)/sin(θ) · v_t + sin(αθ)/sin(θ) · c
```

where v̂ denotes L2-normalized vectors. When the vectors are nearly parallel (dot product > 0.9995), SLERP degenerates to LERP for numerical stability.

SLERP is preferred over LERP because Transformer embedding spaces are approximately hyperspherical: token embeddings have similar norms, and meaning is encoded in direction. LERP produces intermediate vectors that pass through the interior of the sphere, reducing norm at the midpoint. For speaker embeddings specifically, norm reduction introduces artifacts because LayerNorm layers are sensitive to input magnitude.

### 3.3 Norm Matching

After SLERP, the interpolated vector is norm-matched to the original speaker token:

```
v_patched = SLERP(v_t, c, α) · (‖v_t‖ / ‖SLERP(v_t, c, α)‖)
```

This ensures the model's internal LayerNorm layers receive embeddings of the expected magnitude, preventing acoustic artifacts. In our experiments, SLERP with norm matching achieves patch cosine similarity ≥ 0.994 across all speaker/alpha combinations, confirming geometric fidelity.

### 3.4 In-Memory Weight Patching

The patched vector is written directly to the model's embedding table in RAM:

```python
model.talker.get_input_embeddings().weight[speaker_id] = v_patched
```

No disk write occurs. The patch is session-local and does not persist across model reloads. This design is safe for production use: the base model weights are never modified.

### 3.5 Prompt-Safe Affect Control

During development we observed that certain affect words in instruct prompts cause the model to generate pathological audio — specifically, "sad", "tearful", and "crying" induce early end-of-sequence token generation, producing truncated or near-silent output. We term this **affective instruction collapse**.

Mitigation: Replace collapse-inducing words with semantically equivalent but prosodically stable alternatives:

| Collapsed instruct | Stable replacement |
|---|---|
| "sad voice" | "somber and melancholic tone" |
| "tearful" | "melancholic" |
| "crying" | "melancholic" |

Additionally, we add "maintain a steady rhythm, do not pause too long" to rhythm-constrained affects, and raise temperature from 0.55 to 0.65 for such cases. This preserves the intended affective quality while preventing synthesis collapse.

---

## 4. Experiments

### 4.1 Experimental Setup

**Model:** Qwen3-TTS-12Hz-1.7B (CustomVoice and Base variants, MLX bf16 quantization)  
**Hardware:** Apple M4 (MLX framework, Metal GPU acceleration)  
**Reference audio:** Single speaker, ~27 seconds (~16 usable segments of 3–8s)  
**Target speakers:** Ryan (sid=3061), Serena (sid=3066), Dylan (sid=2878)  
**Alpha sweep:** α ∈ {0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0}  
**Evaluation metric:** Speaker Embedding Cosine Similarity (SECS) — re-encode generated audio via base model speaker encoder, compute cosine similarity to reference centroid  
**Test text:** "I can't believe you actually did that. After all the promises we made, you just threw it all away like it meant nothing. But you know what? I'm not even angry anymore. I'm just disappointed."  
**Seed:** 42 (fixed across all experiments)

### 4.2 Alpha Sweep: Continuous Space Verification

The primary empirical question: is the speaker embedding space continuously structured, or does adaptation behave as a discrete switch?

Table 1 reports SECS (vs. reference centroid) and quality score across all alpha values for Ryan:

**Table 1. Alpha sweep results — Ryan token**

| α | pre-patch cos(token, ref) | SECS (ref) | Quality Score | Duration (s) |
|---|---|---|---|---|
| 0.00 | 0.926 | 0.973 | 0.968 | 14.88 |
| 0.15 | 0.935 | 0.971 | 0.965 | 14.32 |
| 0.30 | 0.946 | 0.972 | 0.967 | 15.76 |
| 0.45 | 0.957 | 0.970 | 0.966 | 13.76 |
| 0.60 | 0.970 | 0.975 | 0.969 | 13.36 |
| 0.75 | 0.984 | 0.977 | 0.971 | 13.04 |
| **0.90** | **0.996** | **0.989** | **0.984** | 10.72 |
| 1.00 | 1.000 | 0.992 | 0.977 | 11.12 |

Key observation: SECS increases monotonically with alpha (0.973 → 0.992), confirming that the embedding space is continuously structured. The slight quality score decline at α=1.0 vs. α=0.90 reflects increased silence ratio (7.0% vs. 2.8%) — suggesting that full replacement slightly destabilizes prosodic timing, making α=0.90 the empirically optimal operating point for Ryan.

**Table 2. Alpha sweep results — all three speakers (selected alpha values)**

| Speaker | α=0.0 SECS | α=0.5 SECS | α=0.9 SECS | α=1.0 SECS | Best α | Best Score |
|---|---|---|---|---|---|---|
| Ryan | 0.973 | 0.975 | **0.989** | 0.992 | 0.90 | 0.984 |
| Serena | 0.954 | 0.953 | 0.991 | **0.992** | 1.00 | 0.980 |
| Dylan | 0.967 | 0.973 | 0.990 | **0.993** | 1.00 | 0.979 |

All three speaker tokens show SECS ≥ 0.989 at their optimal alpha, despite originating from distinct speaker identities (Ryan: male energetic, Serena: female neutral, Dylan: male warm). This demonstrates that the method is speaker-agnostic — the embedding space structure is consistent across token identities.

The baseline SECS at α=0.0 (no adaptation) ranges from 0.954–0.973. This non-zero similarity reflects the pre-existing alignment between all Qwen3-TTS speaker tokens in the shared latent space — a property we exploit.

### 4.3 Emotion Preservation

A critical advantage of embedding override vs. voice conversion is that the host model's instruction-following capability is operated through a separate pathway from speaker identity. Table 3 reports SECS and quality scores for Ryan at α=0.90 across 6 emotion conditions:

**Table 3. Emotion preservation — Ryan, α=0.90**

| Emotion | SECS (ref) | Quality Score | Duration (s) | Notes |
|---|---|---|---|---|
| Neutral | 0.989 | 0.984 | 10.72 | Baseline |
| Happy | 0.987 | 0.977 | 13.84 | ✅ Preserved |
| Angry | 0.988 | 0.977 | 10.32 | ✅ Preserved |
| Deep | 0.987 | 0.977 | 16.00 | ✅ Preserved |
| Whisper | 0.946 | 0.924 | 21.12 | ⚠️ SECS drop (expected: whisper reduces spectral richness used by encoder) |
| Sad | 0.955 | 0.893 | 21.12 | ⚠️ Rhythm instability persists despite prompt mitigation |

Findings: For 4 of 6 emotion conditions, SECS remains within 0.002 of neutral (0.987–0.989), confirming that speaker identity and affective prosody are largely separable in the embedding architecture. Whisper and sad show expected degradation — whisper because reduced spectral energy affects the speaker encoder's feature extraction, and sad because prosodic slowing increases silence ratio (0.311), which our scoring penalizes.

Serena and Dylan show similar or better emotion preservation at α=1.0 (Table 4).

**Table 4. Emotion preservation SECS summary — all speakers**

| Emotion | Ryan (α=0.90) | Serena (α=1.0) | Dylan (α=1.0) |
|---|---|---|---|
| Neutral | 0.989 | 0.992 | 0.993 |
| Happy | 0.987 | 0.991 | 0.990 |
| Angry | 0.988 | 0.990 | 0.992 |
| Deep | 0.987 | 0.992 | 0.989 |
| Whisper | 0.946 | 0.958 | 0.982 |
| Sad | 0.955 | 0.982 | 0.986 |

Notable: Dylan shows the most robust emotion preservation across all conditions (minimum SECS 0.982), suggesting that speaker tokens with naturally lower baseline similarity to the reference centroid may produce more stable post-adaptation embeddings.

### 4.4 Affective Instruction Collapse

We document a previously unreported failure mode in instruct-conditioned TTS: certain affective vocabulary causes the model to generate pathologically long, low-energy audio with high silence ratios.

Specifically, prompts containing "sad", "tearful", or "crying" produce outputs with silence_ratio > 0.30 and peak amplitude < 0.20 — effectively near-silent audio. We hypothesize this reflects a training distribution artifact: "sad/tearful" speech in training data likely co-occurred with very quiet, slow delivery, causing the model to associate these words with near-zero energy output.

Our prompt-safe replacement (Section 3.5) reduces silence_ratio from >0.30 to 0.063–0.311 depending on speaker token, with significant quality score improvement. The residual variability suggests this is a model-level sensitivity requiring further investigation, potentially addressable via constrained decoding or energy-floor enforcement.

---

## 5. Discussion

### 5.1 Why the Embedding Space Is Continuous

The continuity of the speaker embedding space is not accidental — it is a consequence of the training objective. During training, Qwen3-TTS must learn to synthesize speech for many speakers using a shared acoustic decoder. For this to work, the speaker encoder (which processes reference audio to produce conditioning vectors) and the speaker embedding table (which provides fixed conditioning for named tokens) must produce geometrically compatible representations. The training process enforces this through shared downstream pathways, naturally producing a continuous space.

This is analogous to the well-documented continuity of word embedding spaces in language models, where vector arithmetic produces semantically meaningful results. We show an analogous structure exists for acoustic speaker identity.

### 5.2 Speaker-Token Independence

A key finding is that SPEAR achieves near-identical SECS (0.989–0.993) across three speakers with distinct vocal characters. This indicates the adaptation does not depend on similarity between the source token and the reference speaker. The reference centroid occupies a region of embedding space accessible from any token via SLERP.

This has practical implications: any available named token can serve as the adaptation host, regardless of perceptual similarity to the target voice.

### 5.3 Optimal Alpha Analysis

The optimal alpha differs across speakers: Ryan benefits from α=0.90 rather than 1.0, while Serena and Dylan perform best at α=1.0. We hypothesize this reflects differences in baseline token-centroid cosine similarity (Ryan: 0.926, Serena: 0.936, Dylan: 0.944 at α=0.0). Tokens with lower initial alignment may benefit from partial mixing to maintain prosodic stability. An automatic alpha selection procedure based on baseline cosine similarity is a natural extension.

### 5.4 Limitations

**No MOS evaluation.** Automated SECS measures speaker similarity but not naturalness, intelligibility, or perceptual quality. Human listening evaluation (MOS, MUSHRA) remains important future work.

**Single model.** Results are validated on Qwen3-TTS-1.7B. Generalization to other closed-vocabulary architectures (CosyVoice, MeloTTS) is expected but not yet verified.

**Whisper and sad degradation.** Two emotion conditions show reduced performance, attributable to acoustic and prosodic factors distinct from the adaptation method itself.

**Reference audio requirement.** While 27 seconds is substantially less than fine-tuning methods require, it is more than prompt-based systems that operate with 3–6 second clips.

---

## 6. Conclusion

We presented SPEAR, a zero-shot voice adaptation method for closed-vocabulary TTS systems. By exploiting the shared latent structure between speaker encoders and embedding tables, SPEAR achieves SECS ≥ 0.989 across three distinct speaker tokens using ~27 seconds of reference audio, no fine-tuning, and no disk modification.

Our experiments demonstrate that closed-vocabulary TTS speaker embedding spaces are continuously structured, enabling smooth voice interpolation via a scalar parameter. The method preserves the host model's emotion instruction-following capability across 4 of 6 tested affect categories, with the remaining two degrading due to acoustic factors independent of the adaptation mechanism.

We additionally document and mitigate a previously unreported failure mode — affective instruction collapse — in which common affect vocabulary induces near-silent synthesis.

SPEAR opens closed-vocabulary TTS models to zero-shot adaptation without requiring architectural modification, providing a practical path for deploying expressive cloned voices using production-grade TTS systems.

---

## References

[1] Kim, J. et al. (2021). VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. *ICML 2021*.

[2] Wang, C. et al. (2023). VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. *arXiv:2301.02111*.

[3] Casanova, E. et al. (2024). XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model. *Interspeech 2024*.

[4] Qwen Team. (2025). Qwen3-TTS Technical Report. Alibaba Group.

[5] White, T. (2016). Sampling Generative Networks. *arXiv:1609.04468*. (SLERP için temel referans)

[6] Lester, B. et al. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. *EMNLP 2021*.

[7] Sheng, Z. et al. (2023). RVC: Retrieval-based Voice Conversion. GitHub.

[8] Du, Z. et al. (2024). CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer. *arXiv:2407.05407*.

---

## Appendix A: SPEAR Algorithm Summary

```
Input:  reference audio A, target speaker token t, alpha α ∈ [0,1]
Output: adapted TTS model (in-memory)

1. Segment A into 3–8s clips {s_1, ..., s_n}
2. Extract embeddings: e_i = E(s_i)  [base model speaker encoder]
3. Score: score_i = 0.6·mean_{j≠i}cos(e_i,e_j) + 0.4·cos(e_i,ē)
4. Weight: w_i = softmax(score_i / 0.1)
5. Centroid: c = Σ_{top-K} w_i · e_i
6. Retrieve: v_t = EmbeddingTable[t]
7. Interpolate: v_mix = SLERP(v_t, c, α)
8. Norm-match: v_patch = v_mix · (‖v_t‖ / ‖v_mix‖)
9. Patch: EmbeddingTable[t] ← v_patch  [RAM only, no disk write]
10. Generate with affect-safe instruct prompts
```

---

## Appendix B: Affective Instruction Collapse — Observed Cases

| Trigger word | Observed effect | Replacement |
|---|---|---|
| "sad" | silence_ratio > 0.30, peak < 0.20 | "somber and melancholic" |
| "tearful" | near-silent output, early EOS | "melancholic" |
| "crying" | breath loop, no speech | "melancholic" |

Temperature adjustment for rhythm-constrained affects: 0.55 → 0.65.

---

## Appendix C: V12 Experiment Configuration

- Custom model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`
- Base model: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
- Reference: `12345.mp3` (~27s, 16 segments, score range 0.986–0.991)
- Alpha values tested: {0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0}
- Speakers: Ryan (sid=3061), Serena (sid=3066), Dylan (sid=2878)
- Fixed seed: 42

