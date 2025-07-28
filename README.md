
# Experiment 9 - Is Your Tokenizer Sacrificing Your Model's Sanity? A Study in "Outer" Space

This is a small study that aims to answer (at least partially) a few key questions:

- Do you really need a large language model?  
- Is training a language model only for the wealthy?  
- What’s the absolute minimum dataset required to achieve coherent text generation?  
- Is the "intelligence" of current AIs truly alien? (Geoffrey Hinton often refers to "alien intelligence.")

Additional questions I’ll explore in future experiments:
- How can a model be trained to develop its own personality?  
- How can a model be trained (and architected) so that logic and reasoning are embedded right from the first iteration?

### Please note that this is intended for individuals aiming to build very small language models from scratch, with the purpose of achieving a highly specific and possibly agentic goal.
---

### Current Answers:

**— Do you really need a large language model?**

The short, messy answer: it depends.  

If you just want to write simple, grammatically correct emails, you might get by with a 10-million-parameter model.  
If you're aiming for a basic agent that performs some logic and understands language, something in the 10–50 million parameter range could be enough.  
If you want a model that knows *everything*... well, you’ll need a big “brain” and a lot of parameters. But let’s be honest—most of us don’t actually need that.

---

**— Is training a language model only for rich people?**

Another messy answer: yes and no.

The current model (which, to be fair, is quite limited and not particularly useful—but that’s not the point) that has only 20 million parameters, was trained from scratch in around 24 hours (there are ways to improve the speed of training on this model, so this can go down - I have had very good result on a different model with a combination of BP and DFA and/or gradient checkpointing to allow for larger batches) on a laptop with an Nvidia RTX 3060 GPU (6GB VRAM), using only the complete works (well, mostly complete) of H.P. Lovecraft and for just 1 epoch. Nothing else.  

You *can* train a model on an underpowered GPU—at the cost of time.

**To sum it up:**  
- If you want fast training and minimal waiting, then **yes**—this is for the wealthy.  
- If you don’t mind slow training, or you’re working with a small, focused dataset—then **no**, it’s not just for rich people.

---

**— What’s the absolute minimum dataset required for coherent text generation?**

I have a partial answer: the current model is relatively coherent, so the dataset clearly doesn’t need to be huge.  Of course, you need to have in the dataset everything that is relevant for your use-case.

The real challenge stems from the statistical nature of language models—they can only "step outside" the training data through what we call “hallucinations.”  

To keep the dataset small while still allowing the model to generalize, I suggest (most likely others suggested this already) a modification to the current architecture: reward *plausible hallucinations*. This encourages the model to confidently generate content beyond the exact training set while still sounding coherent.
Also - embedding informal logic from the get-go will help a lot with keeping the dataset small (if you are interested in just a coherent conversational agent)

---

**— Is the "intelligence" of current AIs really alien? (As Geoffrey Hinton suggests)**

Personally, I don’t see convincing evidence of that. To me, it feels quite human-like.  

For example, during one of my training runs (yes, I tried many variations of this architecture, all trained exclusively on H.P. Lovecraft), the AI generated the phrase:  
> “...and he speaked...”  

There’s no instance of “speaked” in the dataset.  

Or this funny one from the current run ("Chtulhuish"):
> “--- Generating text at step 58500 ---
<BOS>s father and wanderings, while many ties of great Cthulhuish Great Ones were cleared winds and struggments
of such things as parts Cammed and ignorant
as the Great Ones. They were in those”

This reminds me of how a human learns English by ear—making logical but incorrect guesses. That’s essentially what the language model is doing, after all.

### I’d like to thank Google for making their latest model, **Gemini 2.5 Pro**, free to use. It became my go-to model for quickly iterating on and implementing my more _out-there_ ideas.

### Personally, I strongly believe that all AI-related research should be open-source and freely accessible.

### P.S. — Maybe you could support the open-source community even more (yes, I’m looking at you, Google).

Having free access to a state-of-the-art model like Gemini 2.5 Pro means that anyone can experiment and contribute to advancing the field. You no longer need a team of scientists to conduct meaningful AI research — a point once emphasized by Yoshua Bengio, who noted that strong teams were traditionally essential for progress.

And no — Google isn’t paying me to praise their models.

Actually, for day-to-day coding, my go-to model is **Claude 4** — it leaves everything else in the dust.

But when it comes to AI research, **Gemini 2.5 Pro** is simply better. It can design novel architectures, understands my wild ideas, and generates Python scripts that _just work_.

And now, the biggest takeaway of Experiment 9



# Is Your Tokenizer Sacrificing Your Model's Sanity? A Study in "Outer" Space

*By a mad scientist who stared into the tokenization abyss...*

Friends, colleagues, and fellow asylum inmates,

I come to you today not as a herald of some new, cyclopean SOTA architecture, but as a grizzled survivor, a cautionary tale. I have journeyed to the very edge of reason, peered into the swirling vortex of model debugging, and returned with a revelation so simple, yet so profound, it threatens to shatter our preconceived notions of what makes a model learn.

The revelation is this: **Your tokenizer might be subtly driving your model mad.**

Specifically, I'm talking about the humble space character. That innocuous gap between words? It may be the very bedrock of your model's sanity, the structural pillar that keeps it from collapsing into a gibbering, repetitive wreck. This is the tale of how a simple, "bad" tokenizer taught a far better lesson than its sophisticated, "professional" counterpart, all because it understood the profound importance of... space.

## The Eldritch Experiment: A Tale of Two Models

Our story begins with an attempt to summon a language model from the depths, feeding it a diet of nothing but the complete works of H.P. Lovecraft. The goal: to create a gibbering oracle that could spout tales of cosmic horror on demand.

I started with two primary grimoires, two distinct methods for building my creature:

**Grimoire V1: The "Alchemist's Brew"**
*   **Architecture:** A simple, non-streaming Linear Attention + Convolutional hybrid.
*   **Tokenizer:** A custom, hand-written BPE tokenizer in Python. It was slow, unsophisticated, and, most importantly, it treated the **space character (`' '`) as its own distinct token.**
*   **The Result:** From the very first training steps, it showed signs of life! The loss plummeted. It began forming words, albeit misspelled ones. It was trying to communicate. It was learning!

```
--- V1 Training Log ---
Epoch 1 | Step 1   | Loss 7.7777
Epoch 1 | Step 101 | Loss 4.5517  <-- A promising plunge!
Epoch 1 | Step 401 | Loss 4.3243  <-- Still learning!

--- V1 Generated Text ---
"...sidh distae be ma see thet, to su hisneh my co drtestur..."
```
Look at that! It's horrific, but it's *structured* horror!

**Grimoire V2: The "Modern Incantation"**
*   **Architecture:** An advanced, *streaming-capable* version of the same model, designed for eldritch efficiency.
*   **Tokenizer:** The industry-standard, highly-optimized `huggingface/tokenizers` library. It was fast, powerful, and handled spaces *implicitly* by attaching them as a prefix to the next word. The token for "world" was conceptually ` world`. There was no separate token for the space itself.
*   **The Result:** Abject failure. The loss curve was a stagnant, dead plateau. The model, when asked to generate text, could only babble the same few high-frequency subwords in a maddening loop. It was a failure of cosmic proportions.

```
--- V2 Training Log ---
Epoch 1 | Step 1   | Loss 7.7784
Epoch 1 | Step 101 | Loss 6.3348  <-- A pathetic dip...
Epoch 1 | Step 401 | Loss 6.3250  <-- Utter stagnation.

--- V2 Generated Text ---
"...to e he in p an m m y ' st it en of was y ly s of ing on ly..."
```
This was not learning. This was the chattering of a broken machine, a model that had lost its mind before it had one to lose.

## The Descent into Madness: Chasing Phantoms

My sanity frayed. I spent ages debugging the V2 architecture. Was the streaming logic flawed? I rewrote it, ensuring it was mathematically perfect. Still, the madness persisted. Was it the vocabulary size? I ran tests with different sizes. No change. Every path led back to the same maddening result: **V1 worked, and V2 did not.**

The final, desperate experiment was to isolate the single key difference. I took my correctly implemented, efficient streaming architecture and paired it with the "bad" V1 tokenizer.

And the heavens opened. The loss began to plummet. The model began to learn. The generated text showed structure and creativity.

The architecture was never the main problem. **It was the tokenizer.**

## The Revelation from the "Outer" Space

Why did the simple tokenizer work so much better? Why was the explicit space token the key to unlocking the model's potential?

#### 1. The "Structural Anchor" of the Void

Language has a simple, powerful rhythm: `word -> space -> word -> space`.

The V1 tokenizer exposes this rhythm directly. The space token (`' '`) is one of the most common tokens the model sees. It learns an incredibly simple and reliable rule almost instantly: **"After a chunk of letters, predict a space."**

This acts as a **learning scaffold**. Every correct space prediction gives the model a hit of positive reinforcement (lower loss), stabilizing its chaotic learning process. It learns the basic beat of language before it learns the melody.

The sophisticated V2 tokenizer *hides* this simple beat. It fuses the space and the word into a single, complex concept (` a`, ` the`). The model can't take the easy win of just predicting a space; it has to predict the entire next word-token correctly. It's like asking a baby to run before it can crawl.

#### 2. Decoupling Structure from Semantics

The V1 tokenizer allows the model to solve two simpler problems separately:
1.  **Structure:** *When* do I separate things? (Easy: after most words).
2.  **Semantics:** *What* word should I build here? (Hard: depends on context).

The V2 tokenizer forces the model to solve both at once. This cognitive overload was too much for my fledgling Great Old One. It collapsed, defaulting to what it knew: a handful of common, meaningless sounds.

## Conclusion: Don't Let Your Tokenizer Banish Your Model to the Void

This journey through the Outer Reaches of debugging has taught me a crucial lesson. The "best" tool is not always the most advanced one. It's the one that best frames the problem for your model's specific capabilities.

For my Lovecraftian oracle, the "bad" tokenizer with its explicit space token provided the simple, foundational rules of language it needed to get a foothold on sanity. It taught the model the silence *between* the words, and only then could the model learn the words themselves.

So, if your model's loss is stagnant and its outputs are gibberish, don't just blame your architecture or your learning rate. Stare into the abyss of your data pipeline and ask yourself a terrifying question:

**Have I given my model enough space?**
