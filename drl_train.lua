
--[[

This is the script I used to train a RNN based Q-learning network.
RNN(LSTM, GRU) implementation, and this specific script is base on
Andrej Karpathy's char-level RNN text generation program.

]]--
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'

local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Build up a Q-learning module with RNN types of structures.')
cmd:text()
cmd:text('Options')
-- data
-- model params
cmd:option('-rnn_size', 8, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'rnn', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-5,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',.5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
--cmd:option('-clip_delta', 1.0,'max value of Q value change')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))  -- temporarily leave it declaration here. Not sure if I gonna use this type of setting. If off-policy, off-line evaluation is used (e.g., using importance sampling), then this type of mechanism might be needed.
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end


--- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local state_feature_size = 4    -- the dim of input feature set
local action_size = 4   -- output size, should be the # of actions in this rl framework

--- I'm creating a bunch of data here. The data should include information of state, action, instant reward, and terminal, just as it is in the dqn program.
local rl_batch_data_size = 5    -- in total, 200 trajectories.
local rl_max_traj_length = 5   -- max length of transitions in one trajectory
local rl_discount = 0.9
-- For all the following defined rl tensors, the 1st dim is always time index.
-- The tensor rl_states, 1st dim is time index, 2nd is entity index in one batch, 3rd dim is state feature index
rl_states = torch.Tensor{ {{1,0,1,0},{1,1,0,0},{1,1,0,0},{1,1,0,0},{1,0,0,0}},
    {{0,1,1,1},{1,0,0,0},{1,0,0,1},{1,0,0,0},{1,0,0,0}},
    {{0,0,0,0},{1,1,0,0},{0,1,1,1},{0,1,1,1},{1,1,0,0}},
    {{0,0,0,0},{1,0,0,1},{0,0,0,0},{0,0,0,0},{1,0,1,0}},
    {{0,0,0,0},{0,1,1,1},{0,0,0,0},{0,0,0,0},{0,1,1,1}},
    {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}} }     -- the rl_max_traj_length + 1 means in target q calc, an extra state-action q value will be needed. It's not the final format. I need to carefully consider the format of input
rl_actions = torch.Tensor{ {2,4,4,4,1}, {1,4,2,2,3}, {1,4,1,1,3}, {1,2,1,1,2}, {1,1,1,1,1} }
rl_rewards = torch.Tensor{ {0,0,0,0,0}, {-1,0,0,0,0}, {0,0,-1,1,0}, {0,0,0,0,0}, {0,-1,0,0,-1} }
rl_terminals = torch.Tensor{ {0,0,0,0,0}, {1,0,0,0,0}, {1,0,1,1,0}, {1,0,1,1,0}, {1,1,1,1,1}, {1,1,1,1,1} }
--rl_states = torch.Tensor(rl_max_traj_length+1, rl_batch_data_size, state_feature_size):random(1, 100)/100.0     -- the rl_max_traj_length + 1 means in target q calc, an extra state-action q value will be needed. It's not the final format. I need to carefully consider the format of input
--rl_actions = torch.Tensor(rl_max_traj_length, rl_batch_data_size):random(1, action_size)
--rl_rewards = torch.Tensor(rl_max_traj_length, rl_batch_data_size):random(1, 100)/100.0
--rl_terminals = torch.Tensor(rl_max_traj_length+1, rl_batch_data_size):random(0, 1)  -- The +1 has the same meaning as it is for rl_states

-- We also need a preprocessing here. The difference is that, their char_rnn input for each sequence at each time step is one integer (char index),
-- but ours should be a tensor.
-- Todo: pwang8. on Oct20. Change the preprocessing.
-- preprocessing helper function, both params should be of type torch.Tensor
-- Before prepro, both x and y are 2-dim tensors, in which one row contains one sentence, whose length is seq_length, and row # is batch_size.
-- The current prepro() only move the two params onto gpu memory. Transpose will not be used.
function prepro(x)
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    elseif opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
    x = x:cl()
    end
    return x
end

-- When we are going to use a large dataset of training trajectories, this prepro() should be invoked in feval()
if opt.gpuid >= 0 and opt.opencl == 0 then
    rl_states = prepro(rl_states)
    rl_actions = prepro(rl_actions)
    rl_rewards = prepro(rl_rewards)
    rl_terminals = prepro(rl_terminals)
end


-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('Not implemented yet')    -- Todo: pwang8. This should be an important functionality, reloading previous nn model. Need to modify it later.
    os.exit()
--    print('loading a model from checkpoint ' .. opt.init_from)
--    local checkpoint = torch.load(opt.init_from)
--    protos = checkpoint.protos
--    -- make sure the vocabs are the same
--    local vocab_compatible = true
--    local checkpoint_vocab_size = 0
--    for c,i in pairs(checkpoint.vocab) do
--        if not (vocab[c] == i) then
--            vocab_compatible = false
--        end
--        checkpoint_vocab_size = checkpoint_vocab_size + 1
--    end
--    if not (checkpoint_vocab_size == vocab_size) then
--        vocab_compatible = false
--        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
--    end
--    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')   -- If assert fails, the program will quit.
--    -- overwrite model settings based on checkpoint to ensure compatibility
--    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
--    opt.rnn_size = checkpoint.opt.rnn_size
--    opt.num_layers = checkpoint.opt.num_layers
--    opt.model = checkpoint.opt.model
--    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}     -- In the char-rnn program, protos is a table contains 2 entities, the 'rnn' entity and the 'criterion' entity. I guess the torch criterion might not be needed in this rl training.
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(state_feature_size, action_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(state_feature_size, action_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(state_feature_size, action_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
--    protos.criterion = nn.ClassNLLCriterion()   -- It seems like the dqn program does not use a standard criterion module. If it is needed, it should be sth like MSECriterioin.
end

--- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(rl_batch_data_size, opt.rnn_size)    -- batch_size is the # of sequences in one batch. This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
    end
end

--- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

--- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- both params and grad_params are a column vector all parameters, which include all params in this network.
-- They are not put in gpu memory.

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights. Size of bias values equals to 4 * rnn_size for each hidden layer. We have one bias for i, f, o gates and the g (candidate hidden).
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}     -- clones is a table, not torch.Tensor
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, rl_max_traj_length, not proto.parameters)    -- The 2nd param indicates the # of clones for entities of 1st param. This func() implements weight sharing.
end

--for k, v in pairs(clones) do
--    print("In clones table, there exists key: ", k, ', value:', v)
--end


--- Set up target q network
target_protos = {}

function set_target_q_network()
    target_protos = {}
    for k, v in pairs(protos) do
        target_protos[k] = v:clone()
        if opt.gpuid >= 0 and opt.opencl == 0 then
            target_protos[k]:cuda()
        elseif opt.gpuid >=0 and opt.opencl == 1 then
            target_protos[k]:cl()
        end
    end
end

set_target_q_network()


--- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)    -- init_state is actually init hidden/cell state values, zeroed.
function feval(network_param)
    if network_param ~= params then
        params:copy(network_param)  -- set params values to x's values. params contain all trainable parameters in rnn. Don't like their param naming.
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
--    local x, y = loader:next_batch(1)   -- 1 means load next batch from training set. Here the lcoal x is a new variable, different from the x above.
--    x,y = prepro(x,y)   -- this prepro() transpose the tensor of both x and y, exchange their 1st and 2nd dimension.

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global }
    rnn_state[rl_max_traj_length+1] = init_state

    local predictions = {}           -- Q function output, for each action, under certain states
    local loss = 0
    local dloss_dy = {}
    for t=1,rl_max_traj_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{rl_states[t], unpack(rnn_state[t-1])}   -- rl_states[t] is a batch of inputs (state in the rl problem). rnn_state[t-1] is a batch of hidden states (including cell states in lstm) values from the previous time step
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output.--#init_size returns the # of hidden states(including cell states in lstm) in this rnn network. lst[i] has batch_size # of rows and hidden neuron # of columns.
        predictions[t] = lst[#lst] -- last element is the prediction
        -- print('** Hey, ', predictions[t]) -- predictions[t] is a tensor of size 50*65. 50 is batch size. 65 is vocab size. Not bad.


        target_protos.rnn:evaluate()    -- set up the evaluation mode, dropout will be turned off in this mode
        local target_Q = target_protos.rnn:forward{rl_states[t+1], unpack(rnn_state[t])}    -- target_Q is a table contains multiple tensors, including both hidden(cell) states values and output. Output is the last entity in the table.
        local target_Q_max = target_Q[#target_Q]:max(2)    -- the 2nd dim indicates actions. target_Q[#target_Q] is the output tensor. target_Q_max is the Q value of a given state (including) hidden, and maximize over all actions.
        local one_minus_terminal = rl_terminals[t]:clone():mul(-1):add(1)   -- Todo: pwang8. Take care of the rl_terminals index, make sure it is correct.
        local target_Q_value = target_Q_max:clone():mul(rl_discount):cmul(one_minus_terminal)
        local delta = rl_rewards[t]:clone()     -- Todo: pwang8. Check if the index is correct in real application.
        delta:add(target_Q_value)
        local current_Q = torch.Tensor(predictions[t]:size(1))     -- It should be the size of batch_size

        for i=1, predictions[t]:size(1) do
            current_Q[i] = predictions[t][i][rl_actions[t][i]]
        end

        if opt.gpuid >= 0 and opt.opencl == 0 then
            current_Q = current_Q:float():cuda()
            delta = delta:float():cuda()
        end
        delta:add(-1, current_Q)
        loss = loss + delta:pow(2):mul(0.5):cumsum()[rl_states[t]:size(1)] -- this loss here is loss = 0.5 * (y-t)^2
--        delta[delta:ge(opt.clip_delta)] = opt.clip_delta
--        delta[delta:le(-opt.clip_delta)] = -opt.clip_delta


        dloss_dy[t] = torch.zeros(rl_batch_data_size, action_size)

        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            dloss_dy[t] = dloss_dy[t]:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            dloss_dy[t] = dloss_dy[t]:float():cl()
        end

        for i=1,math.min(predictions[t]:size(1), rl_batch_data_size) do     -- Here we are trying to get dloss/dy. We still need to dloss from hidden states calculation to be concatenated together for calculating nn.backward()
            dloss_dy[t][i][rl_actions[t][i]] = delta[i]     -- dloss_dy equals to the gradient d(loss)/d(y) based on mean squre error. Gradient is (y-t) Todo: pwang8. Take care of local declaration.
        end

    end

    loss = loss / (rl_max_traj_length * rl_batch_data_size)

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[rl_max_traj_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=rl_max_traj_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = dloss_dy[t]    -- the return value is d_loss/d_output
        table.insert(drnn_state[t], doutput_t)

        local dlst = clones.rnn[t]:backward({rl_states[t], unpack(rnn_state[t-1])}, drnn_state[t])  -- the 2nd param in module.backward() is dloss/d_output. In rnn, this rnn_output contains real output and hidden state values.
                                                            -- return of this backward() contains dloss/d_input, where input contains both raw input x and previous hidden layer values.
        drnn_state[t-1] = {}
        -- print('*** t:', t, 'dlst size: ', #dlst)
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
                -- print ('# k:', k, 'v:', v)  -- v is 2-dim tensor, size of (batch_size * hidden_neuron_num)
                -- For a one hidden layer rnn/gru, there are 2 entities in dlst. For a sigle hidden layer lstm, this number is 3.
                -- I'll start to guess again. I'm guessing dlst stores gradients of error function wrt x, cell states and hidden states(for lstm).
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    -- grad_params is calculated when model:backward() is invoked
    -- grad_params contains gradient of error function wrt trainable params.
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)    -- clamp() fasten values in the tensor in between this lower and upper bounds.
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = 5000    --opt.max_epochs * loader.ntrain  -- loader.ntrain is the # of batches in training set. opt.max_epochs is the # of epoches to train.
--local iterations_per_epoch = loader.ntrain
local reset_target_q_rate = 100
local loss0 = nil
for i = 1, iterations do
    local epoch = i -- Todo: pwang8. This epoch calculation is not correct now.

    if i % reset_target_q_rate == 1 then
        set_target_q_network()
        print('Reset target q function')
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
        --[[
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        --]]
        cutorch.synchronize()
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % 500 == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

--    -- every now and then or on last iteration
--    if i % opt.eval_val_every == 0 or i == iterations then
--        -- evaluate loss on validation data
--        local val_loss = eval_split(2) -- 2 = validation
--        val_losses[i] = val_loss
--
--        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
--        print('saving checkpoint to ' .. savefile)
--        local checkpoint = {}
--        checkpoint.protos = protos
--        checkpoint.opt = opt
--        checkpoint.train_losses = train_losses
--        checkpoint.val_loss = val_loss
--        checkpoint.val_losses = val_losses
--        checkpoint.i = i
--        checkpoint.epoch = epoch
--        checkpoint.vocab = loader.vocab_mapping
--        torch.save(savefile, checkpoint)
--    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
--    if loss[1] > loss0 * 3 then
--        print('loss is exploding, aborting.')
--        break -- halt
--    end
end


--local rnn_state = {[0] = init_state_global }
--local predictions = {}
--rnn_state[rl_max_traj_length+1] = init_state
--for t=1,rl_max_traj_length do
--    protos.rnn:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
--    local lst = protos.rnn:forward{rl_states[t], unpack(rnn_state[t-1])}   -- rl_states[t] is a batch of inputs (state in the rl problem). rnn_state[t-1] is a batch of hidden states (including cell states in lstm) values from the previous time step
--    rnn_state[t] = {}
--    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output.--#init_size returns the # of hidden states(including cell states in lstm) in this rnn network. lst[i] has batch_size # of rows and hidden neuron # of columns.
--    predictions[t] = lst[#lst] -- last element is the prediction
--end
--
--local trj_ind = 4
--for i=1, rl_max_traj_length do
--    print('At time', i, 'when state is', rl_states[i][trj_ind], 'Q values are', predictions[i][trj_ind])
--end


