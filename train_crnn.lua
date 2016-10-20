
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
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
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
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

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    local checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i) then
            vocab_compatible = false
        end
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    end
    if not (checkpoint_vocab_size == vocab_size) then
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')   -- If assert fails, the program will quit.
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)    -- batch_size is the # of sequences in one batch. This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
    end
end

-- -- Test: init_state is a table, so for each layer, their is a 2-dim matrix, whose row number == batch size (# of sequences in each batch), and column number == # of hidden units in one layer.
-- -- If lstm is used, size of init_state would be doubled. I have no idea about the reason right now. Know it right now. They are used for both hidden state values and cell state values in lstm.
-- -- Initially, all values in init_state table are 0.
-- for i=1, #init_state do
--     print('* key: ' .. i) print(init_state[i]:size())
-- end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- both params and grad_params are a column vector all parameters, which include all params in this network.

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
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
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
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)    -- The 2nd param indicates the # of clones for entities of 1st param.
end

--for k, v in pairs(clones) do
--    print("In clones table, there exists key: ", k, ', value:', v)
--end

-- preprocessing helper function, both params should be of type torch.Tensor
-- Before prepro, both x and y are 2-dim tensors, in which one row contains one sentence, whose length is seq_length, and row # is batch_size.
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous() -- here, x and y should be of type torch.Tensor
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)   -- The 1st param split_index means: 1 - training set, 2 - validation set, 3 - test set. This program here only invokes this function with the 1st param set to 2.
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]   -- This is the number of batches in [split_index] set. In this program here, it is querying # of batches in evaluation set.
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}  -- {[0] = init_state} creats a new table, the entity with index 0 has value of init_state.

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning. -- This sets the mode of the Module (or sub-modules) to train=false. This is useful for modules like Dropout or BatchNormalization that have a different behaviour during training vs evaluation.
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}   -- I guess this is the forward propagate procedure. 1st param is input, 2nd param is previous hidden. The gramma here, in lua, is that it is common to write functions that take a single argument that should be a table. In that case, calling the function does not require use of parenthesis.
            -- lst seems like a table, or tensor. So, I'm curious about what is actually returned from the forward function into lst. Not clear about it right now.
            rnn_state[t] = {}   -- So, this rnn_state variable should store hidden states at each time step. The above line is somehow equal to forward({x[t], unpack(rnn_state[t-1s])})
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end   -- #init_state returns # of hidden layers. lst is the output table from the module. It contains values of hidden state, (cell states, if existed), and output.
            prediction = lst[#lst]    -- I guess hidden state values and output values are both in lst.
            loss = loss + clones.criterion[t]:forward(prediction, y[t])   -- For criterion, the forward function, given an input and a target, compute the loss function associated to the criterion and return the result. In general input and target are Tensors, but some specific criterions might require some other type of object.
            -- Here, the rnn params are clones for many times, so basically, the calculation for each time step in one sequence is the same. And this function is designed for evaluation. So, no param update is needed.
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)    -- init_state is actually init hidden/cell state values, zeroed.
function feval(x)
    if x ~= params then
        params:copy(x)  -- set params values to x's values. params contain all trainable parameters in rnn.
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)   -- 1 means load next batch from training set. Here the lcoal x is a new variable, different from the x above.
    -- print('*** before prepro(), size of x is ', x:size())
    x,y = prepro(x,y)   -- this prepro() transpose the tensor of both x and y, exchange their 1st and 2nd dimension.
    -- print('*** after prepro(), size of x is  ', x:size())  -- after prepro(), row # is # of seq length, column # is batch_size
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}   -- each time, one row of x is sent into model, which contains feature of entities in one batch at a certain time step.
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output. -- #init_state will return # of rows, which is actually batch size. So, rnn_state[t] stores returns of lst from all sequences in one batch.
        predictions[t] = lst[#lst] -- last element is the prediction
        -- print('** Hey, ', predictions[t]) -- predictions[t] is a tensor of size 50*65. 50 is batch size. 65 is vocab size. Not bad.
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t]) -- y[t] is a single integer, which is the target label for this classification problem. I'm feeling this is how target defined in ClassNLLCriterion.
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])    -- the return value is d_loss/d_output
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
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
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
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
local iterations = opt.max_epochs * loader.ntrain  -- loader.ntrain is the # of batches in training set. opt.max_epochs is the # of epoches to train.
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

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
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

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
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end
