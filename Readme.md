# Grl

Applying RNN style of structures in deriving RL policy.

## On Oct 27, 2016, the drl_train.lua is able to run on gpu.Though I'm not pretty clear about how to set its running on gpu perfectly. One main issue is that I'm not sure which specific tensor variable is needed to put into gpu memory. I'll take a look at it later.

Also, very small amount of data will not demonstrate the power of gpu computing. In my current setting, the only 5 trials I got made in one batch (I only have one batch right now) makes cuda 2.5 times slower than cpu.

## On Oct 28, 2016. I've again changed the setting of the rnn/lstm/gru, try to test if deploy it on gpu would finally help. This time I go back to use the totally randomly generated data, and set batch size to 200, input feature number to 30, output neuron size to 30. The trajectory length is still 5 (I guess it should not be parallelly executed). I used 2-layer lstm, with 128 hidden neurons each hidden layer. This time, on the deeplearn work station, gpu outperforms cpu by 3 times. So, it seems like, the advantage of using gpu will be reflected when the network itself is enough complecated. If the NN is small, and batch is small, data dimension is small, then gpu might not have calculation efficiency advantage over cpu.
