require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'

--local image = require 'image'
local Catch = require 'rlenvs.Catch'
local ConvLSTM = require 'model.ConvLSTM'
local model_utils = require 'util.model_utils'

---- Detect QT for image display
--local qt = pcall(require, 'qt')

-- Initialise and start environment
local env = Catch({level = 2, render = true, zoom = 10})
local actionSpace = env:getActionSpace()
local stateSpace = env:getStateSpace()
local game_actions = torch.Tensor(actionSpace['n'])
for aci=1, actionSpace['n'] do
    game_actions[aci] = (aci-1)
end

local reward, terminal
local episodes, totalReward = 0, 0
local batchSize = 50
local rlTrajLength = stateSpace['shape'][3]    -- This is specific to this Catch testbed, because this length is determined by how long the ball could drop donw along the 2nd dimension.

cmd = torch.CmdLine()
cmd:option('-rnn_size', 8, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
cmd:option('-learning_rate',2e-5,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-target_q',1,'set value to 1 means a seperated target Q function is used in training.')

opt = cmd:parse(arg)
convArgs = {}
convArgs.inputDim = stateSpace['shape']     -- input image dimension
convArgs.outputChannel = {3}    -- could have multiple layers
convArgs.filterSize = {2}
convArgs.filterStride = {1}
convArgs.pad = {1}
convArgs.applyPooling = True

--- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
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


-- The following line is commented right now, since qlua cannot find definition of path.exists()
--- make sure output directory exists
--if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end


--- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    -- ConvLSTM model   -- Right now, we are just testing one type of rnn model, which is lstm.
    protos = {}
    protos.rnn = ConvLSTM.convlstm(actionSpace['n'], opt.rnn_size, opt.num_layers, opt.dropout, convArgs)
end

-- graph.dot(protos.rnn.fg, 'MLP', 'outputBasename') -- The generated nn graph has been checked, which seems correct!
print("ConvLSTM Constructed!")

--- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

--- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- both params and grad_params are a column vector all parameters, which include all params in this network.
-- They are not put in gpu memory.

--- initialization of all parameters in the nn
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
--- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
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

--- the initial state of the cell/hidden states, for one batch
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
    end
end

--- Generate the initial hidden/candidate representation for one trajectory, used in trajectory generation, it's one-entity batch
init_state_onetraj = {}
for L=1,opt.num_layers do
    local h_init_traj = torch.zeros(1, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    if opt.gpuid >=0 then h_init_traj = h_init_traj:cuda() end
    table.insert(init_state_onetraj, h_init_traj:clone())
    if opt.model == 'lstm' then
        table.insert(init_state_onetraj, h_init_traj:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
    end
end


print('number of parameters in the model: ' .. params:nElement())
--- make a bunch of clones after flattening, there is one rnn model for each time step, but these models share params
clones = {}     -- clones is a table, not torch.Tensor
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, rlTrajLength, not proto.parameters)    -- The 2nd param indicates the # of clones for entities of 1st param. This func() implements weight sharing.
end

--- Set up target q network
target_protos = {}
function set_target_q_network()
    if opt.target_q == 0 then
        target_protos = protos
    else
        target_protos = {}
        for k, v in pairs(protos) do
            target_protos[k] = v:clone()
            if opt.gpuid >= 0 then
                target_protos[k]:cuda()
            end
        end
    end
end
set_target_q_network()  -- Call it for target Q function initialization

---- Display
--local window = qt and image.display({image = env:start(), zoom=10})   --{image=curr_observ, zoom=10}

--- Use this function to generate trajectories for training
function generate_trajectory()
    -- Here we assume observations are 3-dim images.
    -- In simple RL environments like Catch, rlTrajLength is a fixed number.
    local curr_observ -- This round of game will be dropped from sampling. It is only used for getting an example of observation
    local curr_reward
    local curr_terminal
    -- Construct one batch of observations, actions, rewards, and terminal signals.
    local obs = torch.zeros(rlTrajLength, batchSize, unpack(stateSpace['shape']))  --curr_observ:size()[1], curr_observ:size()[2], curr_observ:size()[3])
    local acts = torch.zeros(rlTrajLength, batchSize, 1)
    local rwds = torch.zeros(rlTrajLength, batchSize, 1)
    local trms = torch.ones(rlTrajLength, batchSize, 1)

    local one_entity_rnn_state = {[0] = init_state_onetraj}    -- only need one set, since each entity in one batch will be conducted serially.
                                                -- No parallelization is set for this data generation step, since we have not
                                                -- use multiple threads to run the atari simulator. Is it possible to use
                                                -- the asychronous methods, like A3C here? That will be interesting to see.


    for ep_iter = 1, batchSize do
        curr_observ = env:start()
        curr_reward = 0
        curr_terminal = 0

        for time_iter = 1, rlTrajLength do
            clones.rnn[time_iter]:evaluate()    -- set to evaluatation mode, turn off dropout
            local one_entity_obs = torch.Tensor(1, curr_observ:size()[1], curr_observ:size()[2], curr_observ:size()[3]) -- A 1-entity sized batch of observation
            one_entity_obs[1] = curr_observ -- Set the current observation to this 1-sized batch. observ is a 3-dim tensor
            local lst = clones.rnn[time_iter]:forward({ one_entity_obs, unpack(one_entity_rnn_state[time_iter-1]) })
            one_entity_rnn_state[time_iter] = {}
            -- add up hidden/candidate states output into the one_entity_rnn_state
            for hid_iter = 1, #init_state_onetraj do table.insert(one_entity_rnn_state[time_iter], lst[hid_iter]) end
            local Q_predict = lst[#lst]
            local act_maxq_index
            _, act_maxq_index = torch.max(Q_predict, 2)     -- 2nd param is 2, meaning to find max value along rows.
            local act_in_env = game_actions[act_maxq_index[1][1]]   -- act_maxq_index is a 2-dim tensor
            -- store these observations into training tensors
            obs[time_iter][ep_iter] = curr_observ
            acts[time_iter][ep_iter] = act_maxq_index   -- Attention: this stored action is the index of that taken action in the output layer of the NN, not the real action # given to the simulator
            rwds[time_iter][ep_iter] = curr_reward
            trms[time_iter][ep_iter] = curr_terminal

            totalReward = totalReward + curr_reward

            env:render()

            if curr_terminal == 1 then
                break
            end
            --- Advance to the next step
            curr_reward, curr_observ, curr_terminal = env:step(act_in_env)  -- act_in_env in rlenvs environment starts its index from 0

            if curr_terminal then
                curr_terminal = 1
            else
                curr_terminal = 0
            end

        end
    end

    episodes = episodes + batchSize

    return obs, acts, rwds, trms
end

--- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)    -- init_state is actually init hidden/cell state values, zeroed.
function feval(network_param)
    if network_param ~= params then
        params:copy(network_param)  -- set params values to x's values. params contain all trainable parameters.
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    --    local x, y = loader:next_batch(1)   -- 1 means load next batch from training set. Here the lcoal x is a new variable, different from the x above.
    --    x,y = prepro(x,y)   -- this prepro() transpose the tensor of both x and y, exchange their 1st and 2nd dimension.

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global }
    rnn_state[rlTrajLength+1] = init_state

    local predictions = {}           -- Q function output, for each action, under certain states
    local loss = 0
    local dloss_dy = {}
    for t=1,rlTrajLength do
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

for i=1, 2 do
    obs, acts, rwds, trms = generate_trajectory()
end

print('Episodes: ' .. episodes)
print('Total Reward: ' .. totalReward)