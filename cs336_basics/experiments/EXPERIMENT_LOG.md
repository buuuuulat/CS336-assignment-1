# Experiment log

> Note: all experiments were made with the same hyperparameters and were trained on 2000 steps.

## Learning rates

- Too small learning rates lead to slow learning speed and requires much more steps to converge
- Too big learning rates lead to unstable learning or divergence
- Optimal learning rate for this experiment is 1e-3

lr = 1e-3
![loss.png](../../runs/lr_experiments/20260824_184917/plots/loss.png)

lr = 1.0
![loss.png](../../runs/lr_experiments/20260824_174325/plots/loss.png)

lr = 1e-4
![loss.png](../../runs/lr_experiments/20260824_194543/plots/loss.png)

## Batch sizes

- Small batch sizes may lead to noisy training does not train models effectively enough
- Very high batch sizes may lead to memory overloads or hurt generalization
- For this case, optimal batch size number is around ~32-128

batch = 1
![loss.png](../../runs/batch_experiments/20260827_140432/plots/loss.png)

batch = 8
![loss.png](../../runs/batch_experiments/20260827_140717/plots/loss.png)

batch = 32
![loss.png](../../runs/lr_experiments/20260824_184917/plots/loss.png)

batch = 48
![loss.png](../../runs/batch_experiments/20260827_142558/plots/loss.png)

## Text generation

**hello, my name is** Sam. Do you want to come with me and I have a spicy pepper!"
"Really?" Tim asks Sam. He has a big mouth and a big smile. "What are you, Sam? Do you want to see it?"
Sam smiles and thinks. He likes spicy food too. He nods and says, "Yes, I would love some spicy pepper so you can see
it."
They put the pepper in Sam's mouth and run to the sink. They wash the pepper with soap and water. The pepper thirsty and
fresh water.
"Yum, this is the spicy pepper," Sam says. "We have to come back now. We have to feed him better."
"Great job, Sam. You are a good friend. I'll take good care of you," Sam says. "Now, you can have some pepper and cheese
to eat."
Sam and Sam smile. They are glad they have each other. They like to play with the spicy pepper. They have a lot of fun
in the garden.
<|endoftext|>

This text is not the greatest, but there is some sort of comprehension. Words are real, some of the sentences make
sense.

## The same model on OpenWebText

![loss.png](../../runs/owt_small_model/plots/loss.png)

Even though the overall loss value is higher on OWT, it is not the greatest idea to rely on it. The loss number also
depends on the vocab size (cross-entropy).

Text generation example:

**hello, my name is** a perfect example of what you’re seeing,” she says. “I like to play with it being a star and a
black card. I’m just one of them. I’m not going to do a thing like that. They have to defend yourself, and they are the
ones that you’re doing. My emotions are not meant to be directed.” “It’s not a thing to you.”

Chanor has a long history and minds, though the various talents of its worlds can be seen. “You have them in a culture
of stuff, they say, ‘Yas, I am going to have to be here.’ This is a good story for me. I’m not going to do it.”

So today, you know, the Star Wars is an American-American novel, and the concept in which we know, as well, are going to
be the most interesting part. But that no such thing has ever been what I am trying to say as a guy that I’ve never
heard of that. I know, it’s likely that when I just think it’s a hero, that’s all that—if I think that there’s another
reason why that’s

## Full OpenWebText run

![loss.png](../../runs/full_runs/full_owt_run/plots/loss.png)
![val_ppl.png](../../runs/full_runs/full_owt_run/plots/val_ppl.png)
![wall_clock.png](../../runs/full_runs/full_owt_run/plots/wall_clock.png)

Model train for about an hour with the following hyperparameters:

```yaml
model:
  vocab_size: 32000
  context_length: 512
  num_layers: 12
  d_model: 768
  num_heads: 12
  d_ff: 2048
  rms_norm_eps: 1.0e-05
  use_causal: true
  use_rope: true
  theta: 10000.0
optimizer:
  name: adamw
  adamw:
    lr: 0.001
    betas:
      - 0.9
      - 0.95
    eps: 1.0e-08
    weight_decay: 0.1
data:
  batch_size: 128
train:
  num_steps: 24000
  max_l2_norm: 1.0
  seed: 42
  lr_schedule:
    t_warmup: 250
    t_c: null
    lr_min_ratio: 0.05
  log_every: 50
  eval_every: 500
  eval_batches: 20
  ckpt_every: 2000
  keep_last: 1
runtime:
  device: cuda
  dtype: bfloat16
```
