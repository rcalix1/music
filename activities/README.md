# 🛠️ Project Tracker

This README helps organize all active and long-term hobbies into a practical weekly and monthly plan. It balances creative energy, technical builds, family fun, and physical health.

---

## 🎯 Guiding Principles

- ✅ Focus on 3–4 active projects max per week
- ⏳ Rotate or defer deep builds monthly
- 🧠 1 big project goal per month
- 🎉 Celebrate something small every week

---

## 🗓️ Weekly Schedule

| Day       | Focus Area             | Activity |
|-----------|------------------------|----------|
| Monday    | 🎵 Piano               | 15–20 min practice (focus on one tricky section) |
| Tuesday   | 🖨️ 3D Printer Projects | Print a telescope or Antikythera part |
| Wednesday | 📻 Radio Kit           | Solder/test for 20 minutes |
| Thursday  | ✈️ RC Plane            | Work on electronics / practice sim |
| Friday    | 🎥 Creative Video       | Record/edit 1-minute build log |
| Saturday  | 🤖 LLM Robot for Daughter | Boot Orin Nano / run basic code |
| Sunday    | 🧠 Reflection + Reading | Zeloof blog + prep tasks for next week |

---

## 🗓️ Monthly Calendar Template

| Week      | Focus Project          | Add-on Task |
|-----------|------------------------|-------------|
| Week 1    | Finish Prusa build     | 3D print working telescope base |
| Week 2    | Radio kit assembly     | Film soldering time-lapse |
| Week 3    | RC plane first flight  | Watch ornithopter videos |
| Week 4    | LLM Robot setup        | Try local speech-to-text demo |

---

## 🧠 Monthly Rotation Plan

| Category               | Item                           | Cycle |
|------------------------|--------------------------------|--------|
| Quick Wins             | Radio kits, solder, test       | Weekly |
| Hardware + Tools       | Prusa, RC plane, ornithopter   | Monthly |
| Long Builds            | LLM bot, 6502, chip research   | Rotating |
| Creative Output        | Piano, Video, Documentation    | Weekly |
| Health & Reflection    | Running, blog reading, photos  | Weekly |

---

## 📆 Example Monthly Calendar (June–July 2025)

### Week 1
- ✅ Finish assembling Prusa 3D printer
- 🖨️ Print telescope part
- 🎵 Piano 2×
- 🧠 Read Zeloof blog on photolithography

### Week 2
- 📻 Complete radio kit #1
- 🎥 Edit 1-min time-lapse of soldering
- 🎵 Learn 8 bars of *Winter*
- 🧠 Plan robot parts for Orin Nano bot

### Week 3
- ✈️ RC plane: complete electronics
- 🕹️ Run RC flight sim
- 🦋 Watch 1 ornithopter flight video
- 🎵 Record short piano clip

### Week 4
- 🤖 LLM buddy bot: audio I/O working
- 💬 Run basic chatbot API
- 🧠 Start first 6502 kit steps (if time)
- 🏃 Run 2× casually

---

## ✅ Done Log (Sample)

| Date       | Task Completed                     | Notes |
|------------|------------------------------------|-------|
| June 29    | Printed telescope gear             | Looks smooth, reprint with tighter tolerances |
| June 27    | Piano: nailed Vivaldi transition   | 2× 20 min sessions helped a lot |
| June 26    | Soldered half the radio kit        | Ready for power test |

---

## 🧠 Weekly Review Prompts

- What did I actually build this week?
- What small thing brought joy?
- What’s the next micro-step?
- Did I spend time with my daughter in a fun/tech way?
- Can I document something in 60 seconds of video?

---

> ✨ *You're building more than projects. You're building a lifestyle of creativity and curiosity.*

---

# AI Research Portfolio README

This document summarizes 3 strategically aligned research projects, grounded in real systems (blast furnace, energy data, RC flight), and infused with cutting-edge AI methods (preference learning, latent modeling, self-critique, and autonomous control).

![Projects Diagram](Three_Projs.png)


---

## 🥇 Project 1: Preference-Based Time Series GPTs

**Theme**: Fine-tune GPT-style models for time series forecasting using preference-based methods instead of traditional supervision.

### Key Concepts

* **DPO** (Direct Preference Optimization)
* **GRPO** (Gradient Reward Preference Optimization)
* **SPO** (Score-based Preference Optimization – proposed)
* Preference signals from score metrics (R^2, RMSE) or human annotations

### Applications

* Blast furnace silicon prediction
* UCI Appliances Energy dataset

### Inspirations

* *“Direct Preference Optimization: Your Language Model is Secretly a Reward Model”* by Rafailov et al. (2023)
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

* *"DeepSeek Math"
  https://arxiv.org/abs/2402.03300

* *“RLHF: Reinforcement Learning with Human Feedback”* (OpenAI summary + implementation base)
  [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

### Goals

* Replace labeled regression losses with preference-based loops
* Fine-tune GPTs with human or score-derived preferences
* Extend to self-refining loops with critic models (see Project 3)

---

## 🛫 Project 2: Sim2Real Flight Agents for Autonomous Control

**Theme**: Extend prior RL-based autopilot research using simulation (X-Plane 11) into modern, preference-aware agents deployable to physical RC aircraft.

### Key Concepts

* Simulation-to-reality transfer (Sim2Real)
* Curriculum learning + preference signals
* GameNGen-style evaluation environments

### Hardware + Simulation

* X-Plane 11 (used in prior publications)
* RC planes with onboard electronics

### Inspirations

* *“Dreamer: Scalable Reinforcement Learning with World Models”* by Hafner et al. (2020) https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/ 
  [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)

* *GameNGen: General Game Engine for Curriculum RL Training* ([https://gamengen.github.io](https://gamengen.github.io))

* *“Curriculum Learning”* by Bengio et al. (2009)
  https://dl.acm.org/doi/10.1145/1553374.1553380

* *"CHM MAKES ALEXNET SOURCE CODE AVAILABLE TO THE PUBLIC"
https://computerhistory.org/press-releases/chm-makes-alexnet-source-code-available-to-the-public/

* *"Stable Diffusion Examples"
https://github.com/rcalix1/LatentDiffusion/tree/main

* *"Kestrel"
https://github.com/rcalix1/robotics/tree/main/ornithopters/hardware/kestrel
  
### Goals

* Rebuild flight agents in Gym-like environment
* Incorporate preference scores or critiques for flight behaviors
* Transfer to RC hardware with modular policy design

---

## 🤖 Project 3: Forecast Critique and Correction Loop

**Theme**: Pair a forecasting GPT with a learned Critic that scores the forecast’s quality (e.g., plausibility, expected RMSE), enabling revision and self-correction.

### Key Concepts

* Self-rating and self-improving models
* Critic model trained from score metrics
* Optional forecast refinement step

### Inspirations

* *“Self-Refine: Iterative Refinement with Self-Feedback”* by Madaan et al. (2023)
  [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)

* *“Chain-of-Verification Reduces Hallucination in Large Language Models", Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston
(https://arxiv.org/abs/2309.11495)

* *“RLAIF: Reinforcement Learning from AI Feedback”* by Bai et al. (Anthropic, 2023)
  [[arXiv:2307.15217](https://arxiv.org/abs/2307.15217)](https://cameronrwolfe.substack.com/p/rlaif-reinforcement-learning-from)

* *"Unexpectedness as a Measure of Semantic Learning"
https://github.com/rcalix1/music/blob/main/activities/IJAITsemanticUnexpectednessTransformer.pdf

### Goals

* Create a GPT-based forecasting system with feedback loop
* Train a Critic model to evaluate or flag poor forecasts
* Enable auto-revision or ranking of outputs

---


### Domains

* UCI Energy Prediction dataset 
* Blast furnace time series
* Latent dynamics for RC plane or control logs

### Goals

* Predict future latent states rather than raw sensor values
* Learn internal representations that minimize surprise
* Move toward agents that simulate and reason with latent futures

---




