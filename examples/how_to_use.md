# How to Use CoMLRL

CoMLRL is designed for LLM collaboration with multiple MARL algorithms. To set up a training script, you need to have:

## Reward Model

A reward model can be a function that takes all agents' completions and gives a reward. It can also be a list of multiple reward functions where the total rewards are calculated with reward weights. The total reward can be transformed by a predefined reward processor (some are provided in utils/reward_processors).

```
todo
```

## Dataset

you can design your own dataset like ... or a simple way is to use huggingface portal by just giving a string

### Configuration and Trainer

what are the necessary configurations?

what are given to trainer?
