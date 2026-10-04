# AI Engineering Hub — Directory Structure Map

## Visual Organization

```
ai_engineering/
│
├── 📄 README.md                          ← START HERE: Main navigation hub
│
├── 📚 notes/                             Pure learning & theory
│   ├── README.md                         Notes index
│   ├── session-notes.md                  General session notes
│   │
│   ├── fundamentals/                     Core AI concepts
│   │   ├── README.md
│   │   ├── AI_Engineering_Intro_Notes.md
│   │   ├── transformer-internals.md
│   │   ├── backprop_ninja_notes.md
│   │   └── backprop-ninja-variables.md
│   │
│   └── specializations/                  Advanced topics
│       ├── README.md
│       ├── jev_explained.md
│       ├── finetuning_lora_qlora_guide.md
│       └── ml-research-engineer-path.md
│
├── 🚀 projects/                          Working applications & demos
│   ├── README.md                         Projects overview & guide
│   │
│   ├── demo/                             ReAct agent demo (Streamlit)
│   │   ├── README.md
│   │   ├── app.py
│   │   ├── agent/
│   │   ├── gitlab/
│   │   └── ui/
│   │
│   ├── microgpt/                         GPT from scratch (Python + TS)
│   │   ├── README.md
│   │   ├── microgpt.py
│   │   ├── microgpt.ts
│   │   ├── learning-micro-gpt.md
│   │   ├── training-notes.md
│   │   └── inference-notes.md
│   │
│   ├── neural-net/                       Network learning visualization
│   │   ├── README.md
│   │   ├── nn-learns/
│   │   │   ├── app.py
│   │   │   ├── ui.py
│   │   │   └── core/
│   │   └── how-a-network-learns-streamlit-script.md
│   │
│   └── inference-machine-demo/           Inference simulator (HTML)
│       ├── README.md
│       ├── index.html
│       ├── inference-machine-talk-script.md
│       └── inference-machine-demo-spec.md
│
├── 🎓 learning-resources/                Courses & formal content
│   ├── README.md
│   │
│   └── CS229/                            Stanford ML course
│       ├── README.md
│       ├── CS229-linear-regression-notes.md
│       ├── CS229-lecture3-notes.md
│       ├── session-notes.md
│       └── assets/
│
└── 🎨 assets/                            Supporting materials
    ├── README.md
    ├── backprop_ninja_assets/
    ├── fine_tuning_assets/
    └── jev_assets/
```

## Quick Navigation Guide

### 🎯 By Purpose

**Want to learn?** 
→ [notes/](./notes/README.md)
- Start: [fundamentals/](./notes/fundamentals/)
- Then: [specializations/](./notes/specializations/)

**Want to build/explore?**
→ [projects/](./projects/README.md)
- Quick demo: [demo/](./projects/demo/)
- Implementation: [microgpt/](./projects/microgpt/)
- Visualization: [neural-net/](./projects/neural-net/)

**Want structured coursework?**
→ [learning-resources/](./learning-resources/README.md)
- [CS229/](./learning-resources/CS229/)

**Looking for diagrams/visuals?**
→ [assets/](./assets/README.md)

### 📚 By Topic

| Topic | Files | Projects |
|-------|-------|----------|
| **Fundamentals** | AI Intro, Transformers, Backprop | neural-net, microgpt |
| **Implementation** | GPT building, autograd, inference | microgpt |
| **Fine-tuning** | LoRA, QLoRA guide | demo (optional extension) |
| **Architecture** | Transformer internals, MLP | microgpt, neural-net |
| **Training** | SGD, Adam, cross-entropy, loss | microgpt |
| **ML Theory** | Linear regression, MLEs | CS229 |
| **Inference** | Sampling, temperature, KV cache | microgpt, inference-machine-demo |
| **Agents** | ReAct pattern, tools, memory | demo |

### 🚀 Getting Started Paths

**Path 1: Beginner → Fundamentals**
1. Read: [notes/fundamentals/AI_Engineering_Intro_Notes.md](./notes/fundamentals/AI_Engineering_Intro_Notes.md)
2. Read: [notes/fundamentals/transformer-internals.md](./notes/fundamentals/transformer-internals.md)
3. Explore: [projects/neural-net/](./projects/neural-net/) (visual learning)

**Path 2: Deep Dive → Implementation**
1. Read: [notes/fundamentals/backprop_ninja_notes.md](./notes/fundamentals/backprop_ninja_notes.md)
2. Study: [projects/microgpt/](./projects/microgpt/) README and notes
3. Review: [projects/neural-net/](./projects/neural-net/) for intuition

**Path 3: Advanced → Specialization**
1. Read: [notes/specializations/finetuning_lora_qlora_guide.md](./notes/specializations/finetuning_lora_qlora_guide.md)
2. Review: [notes/specializations/jev_explained.md](./notes/specializations/jev_explained.md)
3. Check: [notes/specializations/ml-research-engineer-path.md](./notes/specializations/ml-research-engineer-path.md)

**Path 4: Hands-On → Building & Demoing**
1. Run: [projects/demo/](./projects/demo/) (ReAct agent demo)
2. Explore: [projects/inference-machine-demo/](./projects/inference-machine-demo/) (inference visualization)
3. Study: [projects/microgpt/](./projects/microgpt/) (implementation details)

## File Statistics

| Category | Count | Notes |
|----------|-------|-------|
| **Notes** | 8 | Fundamentals (4) + Specializations (3) + Session (1) |
| **Projects** | 4 | All runnable/interactive |
| **Courses** | 1 | CS229 with 3 materials |
| **README files** | 12 | Comprehensive navigation |
| **Assets folders** | 3 | Consolidated visuals |

## Recent Changes

- ✅ Reorganized notes into fundamentals vs specializations
- ✅ Moved projects to dedicated folder with individual READMEs
- ✅ Consolidated assets folder (removed scattered asset directories)
- ✅ Created comprehensive navigation READMEs at each level
- ✅ Flattened nested folder structures (inference-machine-demo)
- ✅ Created this visual map

---

**Last updated:** 2026-10-04

← [Back to README](./README.md)
