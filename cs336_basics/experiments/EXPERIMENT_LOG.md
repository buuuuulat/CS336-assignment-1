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
Sam smiles and thinks. He likes spicy food too. He nods and says, "Yes, I would love some spicy pepper so you can see it."
They put the pepper in Sam's mouth and run to the sink. They wash the pepper with soap and water. The pepper thirsty and fresh water.
"Yum, this is the spicy pepper," Sam says. "We have to come back now. We have to feed him better."
"Great job, Sam. You are a good friend. I'll take good care of you," Sam says. "Now, you can have some pepper and cheese to eat."
Sam and Sam smile. They are glad they have each other. They like to play with the spicy pepper. They have a lot of fun in the garden.
<|endoftext|>

This text is not the greatest, but there is some sort of comprehension. Words are real, some of the sentences make sense.


## RMSNorm ablation

![loss.png](../../runs/rmsnorm/rmsnorm_ablation/plots/loss.png)
Without RMSNorm, training becomes much more unstable. Spikes caused by overfitting can also be seen.


## Post-norm transformer
Post norm learning curve
![loss.png](../../runs/rmsnorm/post_norm/plots/loss.png)

Pre norm learning curve
![loss.png](../../runs/lr_experiments/20260824_184917/plots/loss.png)

Even though the results look the same, RMSNorm gives a dramatic boost in convergence speed and stability, especially
the pre-norm version of it.

