# Grl

Applying RNN style of structures in deriving RL policy.

## On Oct 27, 2016, the drl_train.lua is able to run on gpu.Though I'm not pretty clear about how to set its running on gpu perfectly. One main issue is that I'm not sure which specific tensor variable is needed to put into gpu memory. I'll take a look at it later.

Also, very small amount of data will not demonstrate the power of gpu computing. In my current setting, the only 5 trials I got made in one batch (I only have one batch right now) makes cuda 2.5 times slower than cpu.
