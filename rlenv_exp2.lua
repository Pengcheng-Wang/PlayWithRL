--
-- User: pwang8
-- Date: 12/22/16
-- Time: 10:52 AM
-- This script is modified from rlenv_exp1.lua
-- The main purpose of this script is to train an ConvLSTM model
-- to play a simple game.
-- In this program, I applied 3-time sized training blocks, in which one block will all positive trajectories,
-- one with all negative trajectories, and one with sampled running trajectories.
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'

cmd = torch.CmdLine()
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
cmd:option('-learning_rate',2e-7,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',2000,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_freq',2000,'frequency of learning rate decay, in number of epochs')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',30,'number of sequences to train on in parallel')
cmd:option('-batch_block',3,'number of batch blocks in training tensor.')
cmd:option('-max_epochs',100000,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-target_q',2000,'The frequency to update target Q network. Set it to 0 if target Q is not needed.')
cmd:option('-rl_discount', 0.99, 'Discount factor in reinforcement learning environment.')
cmd:option('-clip_delta', 1, 'Clip delta in Q updating.')
cmd:option('-L2_weight', 2e-3, 'Weight of derivative of L2 norm item.')
cmd:option('-greedy_ep_start', 1.0, 'The starting value of epsilon in ep-greedy.')
cmd:option('-greedy_ep_end', 0.1, 'The ending value of epsilon in ep-greedy.')
cmd:option('-greedy_ep_startEpisode', 1, 'Starting point of training and epsilon greedy sampling.')
cmd:option('-greedy_ep_endEpisode', 30000, 'End point of training and epsilon greedy sampling.')
cmd:option('-write_every', 200, 'Frequency of writing models into files.')
cmd:option('-train_count', 1, 'Number of trainings conducted after each sampling.')
cmd:option('-RL_env', 'rlenvs.Catch', 'The name of rlenv environment.')
cmd:option('-game_level', 4, 'The difficulty level of the game.')
cmd:option('-traj_length', 0, 'The max trajectory length in training an RNN. Set it to 0 if traj length could be calculated')
cmd:option('-convnet_set', 'convnet_rlenv1', 'The CNN layers (under RNN) setting.')
cmd:option('-print_freq',50,'frequency of printing result on screen')
cmd:option('-gc_freq',50,'frequency of invoking garbage collection')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local _, RlenvGame = pcall(require, opt.RL_env)
local ConvLSTM = require 'model.ConvLSTM'
local model_utils = require 'util.model_utils'

--- Initialise and start environment
local env = RlenvGame({level = opt.game_level, render = true, zoom = 10})
local actionSpace = env:getActionSpace()
local num_actions = actionSpace['n']    -- number of optinal actions in this environment
local stateSpace = env:getStateSpace()
local game_actions = torch.Tensor(num_actions)
for aci=1, num_actions do
    game_actions[aci] = (aci-1)
end

--- The Convolution Layer Setting
convArgs = {}
convArgs.inputDim = stateSpace['shape']     -- input image dimension
local _, convSet = pcall(require, opt.convnet_set)
convSet(convArgs)   -- Call the function() in convnet_rlenv1

local reward, terminal
local episodes, totalReward = 0, 0
local rlTrajLength = stateSpace['shape'][3]    -- This is specific to this Catch testbed, because this length is determined by how long the ball could drop donw along the 2nd dimension.
if opt.traj_length > 0 then rlTrajLength = opt.traj_length end  -- If it is indicated by user, use it.

local batchSize = opt.batch_size
local sample_iter = 1

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
    opt.dropout = checkpoint.opt.dropout
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    -- ConvLSTM model
    protos = {}
    protos.rnn = ConvLSTM.convlstm(num_actions, opt.rnn_size, opt.num_layers, opt.dropout, convArgs)
end

-- graph.dot(protos.rnn.fg, 'MLP', 'outputBasename') -- The generated nn graph has been checked, which seems correct!
print("ConvLSTM Constructed!")

--- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:float():cuda() end
end

--- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- both params and grad_params are a column vector all parameters, which include all params in this network.
-- They are not put in gpu memory.

--- initialization of all parameters in the nn
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

--- Below is initialization using either He et al., 2015 method or Xavier init method
-- Note: right now, using this init method will lead to larger param init value than uniform(-0.08, 0.08)

--local init_conv_width = stateSpace['shape'][2]
--local init_conv_height = stateSpace['shape'][3]
for _,node in ipairs(protos.rnn.forwardnodes) do
    if node.data.module and node.data.module.weight then
        local fanin = node.data.module.weight:size(2)
        local fanout = node.data.module.weight:size(1)
        -- For ReLU Conv layer, use He et al., 2015 init method
        if torch.type(node.data.module) == 'nn.SpatialConvolution' then
            -- These following calculation will be useful if Xavier init is used in Conv Layers.
            --            --- This calculation should be correct if max polling is not applied
            --            fanin = node.data.module.nInputPlane * init_conv_width * init_conv_height
            --            init_conv_width = (init_conv_width - node.data.module.kW) / node.data.module.dW + 1
            --            init_conv_height = (init_conv_height - node.data.module.kH) / node.data.module.dH + 1
            --            fanout = node.data.module.nOutputPlane * init_conv_width * init_conv_height
            -- Right now, we try to use the ReLU CNN init method proposed by Kaiming He etc. in theirICCV 2015 paper.
            node.data.module.weight:normal(0, math.sqrt(2 / (node.data.module.kW * node.data.module.kH * fanout)))
        else
            -- otherwise, for other types of layers, use Xavier initialization
            local uni_dist_length = math.sqrt(6 / (fanin + fanout))
            node.data.module.weight:uniform(-1*uni_dist_length, uni_dist_length)
        end
    end
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
init_state_onetraj = {}
init_state_onetraj_cpu = {}
for L=1,opt.num_layers do
    local h_init
    if opt.batch_block > 0 then
        h_init = torch.zeros(batchSize * 3, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    else
        h_init = torch.zeros(batchSize, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    end
    local h_init_traj = torch.zeros(1, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    local h_init_traj_cpu = torch.zeros(1, opt.rnn_size)    -- This table init_state has the dimension of (# of seqs * # of hidden neurons). So, this table should be used to store the hidden layer value in RNN/GRU, and both cell state and hidden state values in LSTM at previous time step (if it is not only used to represent the initial hidden/cell layer states). This is the reason why LSTM has doubled memory space for init_state.
    if opt.gpuid >=0 then
        h_init = h_init:float():cuda()
        h_init_traj = h_init_traj:float():cuda()
    end
    table.insert(init_state, h_init:clone())
    table.insert(init_state_onetraj, h_init_traj:clone())
    table.insert(init_state_onetraj_cpu, h_init_traj_cpu:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
        table.insert(init_state_onetraj, h_init_traj:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
        table.insert(init_state_onetraj_cpu, h_init_traj_cpu:clone())    -- This table init_state is used to store hidden state and cell state values. So, for LSTM, it requires doubled space, for both storing s and c values. The number of lines indicates all sequences in one batch could be processed parallelly.
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
                target_protos[k]:float():cuda()
            end
        end
    end
end
set_target_q_network()  -- Call it for target Q function initialization


local obs_train
local acts_train
local rwds_train
local trms_train

if opt.batch_block > 0 then
    -- if batch_block mode is on, all tensors will be tripled to contain
    -- normal observation, all positive observation, and all negative observations.
    obs_train = torch.zeros(rlTrajLength, batchSize * 3, unpack(stateSpace['shape']))  --curr_observ:size()[1], curr_observ:size()[2], curr_observ:size()[3])
    acts_train = torch.LongTensor(rlTrajLength, batchSize * 3, 1):fill(1) -- Attention: the stored action is the index of those actions in the output layer of the NN, not the real action # sent to the simulator
    rwds_train = torch.zeros(rlTrajLength, batchSize * 3, 1)
    trms_train = torch.ByteTensor(rlTrajLength, batchSize * 3, 1):fill(1)
else
    -- Otherwise, we only store normal observation.
    obs_train = torch.zeros(rlTrajLength, batchSize, unpack(stateSpace['shape']))  --curr_observ:size()[1], curr_observ:size()[2], curr_observ:size()[3])
    acts_train = torch.LongTensor(rlTrajLength, batchSize, 1):fill(1) -- Attention: the stored action is the index of those actions in the output layer of the NN, not the real action # sent to the simulator
    rwds_train = torch.zeros(rlTrajLength, batchSize, 1)
    trms_train = torch.ByteTensor(rlTrajLength, batchSize, 1):fill(1)
end

-- These variables will be used to help fill in pos/neg batch blocks
-- The random observations store from index 1 to batchSize
-- Positive trajectories store form index batchSize+1 to 2*batchSize
-- Negative trajectories store form index 2*batchSize+1 to 3*batchSize
local batch_pos_block_iter = batchSize + 1
local batch_neg_block_iter = 2 * batchSize + 1
local batch_pos_block_full = false
local batch_neg_block_full = false
--- Use this function to generate trajectories for training
function generate_trajectory()
    -- Here we assume observations are 3-dim images.
    -- In simple RL environments like Catch, rlTrajLength is a fixed number.
    local curr_observ
    local curr_reward
    local curr_terminal
    -- Construct one batch of observations, actions, rewards, and terminal signals.
    local obs = obs_train
    local acts = acts_train
    local rwds = rwds_train
    local trms = trms_train

    local one_entity_rnn_state = {[0] = init_state_onetraj_cpu}    -- only need one entity(trajectory), since each entity in one batch will be conducted serially.
    -- No parallelization is set for this data generation step, since we have not
    -- used multiple threads to run the atari simulator. Is it possible to use
    -- the asychronous methods, like A3C here? That will be interesting to see.
    curr_observ = env:start()
    curr_reward = 0
    curr_terminal = 0   -- We assume each trajectory will not terminate instantly after initialization

    local cpu_proto_smpl = {}
    if opt.gpuid >= 0 then
        for name,proto in pairs(protos) do
            cpu_proto_smpl[name] = proto:clone():double()
        end
    else
        cpu_proto_smpl = protos
    end
    cpu_proto_smpl.rnn:evaluate()    -- set to evaluatation mode, turn off dropout

    local ep_iter = sample_iter % batchSize
    if ep_iter == 0 then ep_iter = batchSize end

    for time_iter = 1, rlTrajLength do
        local one_entity_obs = torch.Tensor(1, curr_observ:size()[1], curr_observ:size()[2], curr_observ:size()[3]) -- A 1-entity sized batch of observation
        one_entity_obs[1] = curr_observ -- Set the current observation to this 1-sized batch. observ is a 3-dim tensor
        local lst = cpu_proto_smpl.rnn:forward({ one_entity_obs, unpack(one_entity_rnn_state[time_iter-1]) })
        one_entity_rnn_state[time_iter] = {}
        -- add up hidden/candidate states output into the one_entity_rnn_state
        for hid_iter = 1, #init_state_onetraj do table.insert(one_entity_rnn_state[time_iter], lst[hid_iter]) end
        local Q_predict = lst[#lst]
        local act_maxq_index
        _, act_maxq_index = torch.max(Q_predict, 2)     -- 2nd param is 2, meaning to find max value along rows.

        --- epsilon-greedy
        local greedy_ep = (opt.greedy_ep_end + math.max(0, (opt.greedy_ep_start - opt.greedy_ep_end) *
                (opt.greedy_ep_endEpisode - math.max(0, sample_iter - opt.greedy_ep_startEpisode)) / opt.greedy_ep_endEpisode))
        -- Epsilon greedy
        if torch.uniform() < greedy_ep then
            act_maxq_index[1][1] = torch.random(1, num_actions)
        end

        local act_in_env = game_actions[act_maxq_index[1][1]]   -- act_maxq_index is a 2-dim tensor

        -- store these observations into training tensors
        if opt.gpuid >= 0 then
            obs[time_iter][ep_iter] = curr_observ:clone():float():cuda()
            acts[time_iter][ep_iter] = act_maxq_index:clone():float():cuda()   -- Attention: this stored action is the index of that taken action in the output layer of the NN, not the real action # given to the simulator
            rwds[time_iter][ep_iter] = curr_reward
            trms[time_iter][ep_iter] = curr_terminal
        else
            obs[time_iter][ep_iter] = curr_observ
            acts[time_iter][ep_iter] = act_maxq_index   -- Attention: this stored action is the index of that taken action in the output layer of the NN, not the real action # given to the simulator
            rwds[time_iter][ep_iter] = curr_reward
            trms[time_iter][ep_iter] = curr_terminal
        end

        totalReward = totalReward + curr_reward

        env:render()

        if curr_terminal == 1 then
            -- We assume the rl environment setting is as: given an numerical reward when ternimal state is reached
            if curr_reward > 0 then
                -- If positive reward signal is given, copy this trajectory to the 2nd block
                for pos_block_time_iter = 1, time_iter do
                    obs[pos_block_time_iter][batch_pos_block_iter] = obs[pos_block_time_iter][ep_iter]:clone()
                    acts[pos_block_time_iter][batch_pos_block_iter] = acts[pos_block_time_iter][ep_iter]:clone()
                    rwds[pos_block_time_iter][batch_pos_block_iter] = rwds[pos_block_time_iter][ep_iter]:clone()
                    trms[pos_block_time_iter][batch_pos_block_iter] = trms[pos_block_time_iter][ep_iter]:clone()
                end
                batch_pos_block_iter = batch_pos_block_iter + 1
                if batch_pos_block_iter > 2 * batchSize then
                    batch_pos_block_iter = batchSize + 1
                    batch_pos_block_full = true
                end
            else
                -- otherwise, copy this trajectory to the 3rd block
                for neg_block_time_iter = 1, time_iter do
                    obs[neg_block_time_iter][batch_neg_block_iter] = obs[neg_block_time_iter][ep_iter]:clone()
                    acts[neg_block_time_iter][batch_neg_block_iter] = acts[neg_block_time_iter][ep_iter]:clone()
                    rwds[neg_block_time_iter][batch_neg_block_iter] = rwds[neg_block_time_iter][ep_iter]:clone()
                    trms[neg_block_time_iter][batch_neg_block_iter] = trms[neg_block_time_iter][ep_iter]:clone()
                end
                batch_neg_block_iter = batch_neg_block_iter + 1
                if batch_neg_block_iter > 3 * batchSize then
                    batch_neg_block_iter = 2 * batchSize + 1
                    batch_neg_block_full = true
                end
            end

            break
        end
        --- Advance to the next step
        curr_reward, curr_observ, curr_terminal = env:step(act_in_env)  -- act_in_env in rlenvs environment starts its index from 0

        if curr_terminal then
            curr_terminal = 1
            -- Here, I made some changes which modified the rlenv setting.
            if curr_reward == 0 then
                curr_reward = -1    -- If game terminates, and play does not succeed, then give reward of -1.
            end
        else
            curr_terminal = 0
        end

    end

    episodes = episodes + 1
end


--- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)    -- init_state is actually init hidden/cell state values, zeroed.
function feval(network_param)
    if network_param ~= params then
        params:copy(network_param)  -- set params values to x's values. params contain all trainable parameters.
    end
    grad_params:zero()

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global }

    local predict_Q_values = {}           -- Q function output, for each action, under certain states
    local loss = 0
    local dloss_dy = {}
    for t=1, rlTrajLength-1 do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward({ obs_train[t], unpack(rnn_state[t-1]) })    --{rl_states[t], unpack(rnn_state[t-1])}   -- rl_states[t] is a batch of inputs (state in the rl problem). rnn_state[t-1] is a batch of hidden states (including cell states in lstm) values from the previous time step
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output.--#init_size returns the # of hidden states(including cell states in lstm) in this rnn network. lst[i] has batch_size # of rows and hidden neuron # of columns.
        predict_Q_values[t] = lst[#lst] -- last element is the prediction

        target_protos.rnn:evaluate()    -- set up the evaluation mode, dropout will be turned off in this mode
        local target_Q = target_protos.rnn:forward{obs_train[t+1], unpack(rnn_state[t])}    -- target_Q is a table contains multiple tensors, including both hidden(cell) states values and output. Output is the last entity in the table.
        local target_Q_max = target_Q[#target_Q]:max(2)    -- the 2nd dim indicates find max value along row. target_Q[#target_Q] is the output tensor. target_Q_max is the Q value of a given state (including) hidden, and maximize over all actions.
        local one_minus_terminal
        if opt.gpuid >= 0 then
            one_minus_terminal = trms_train[t+1]:clone():mul(-1):add(1):float():cuda()
        else
            one_minus_terminal = trms_train[t+1]:clone():mul(-1):add(1):double()
        end
        local target_Q_value = target_Q_max:clone():mul(opt.rl_discount):cmul(one_minus_terminal)
        local delta = rwds_train[t+1]:clone()
        delta:add(target_Q_value)   -- delta == reward + (1-terminal) * discount * Q_max_a(s_t+1, a_t+1). delta is a 2-dim tensor.

        local current_Q = torch.Tensor(predict_Q_values[t]:size(1), 1)     -- It should be the size of batch_size

        for i=1, predict_Q_values[t]:size(1) do     -- for each entity in one batch
        current_Q[i] = predict_Q_values[t][i][acts_train[t][i][1]]     -- Get Q values for specific actions taken by agent
        end                                         -- current_Q is a 1-dim tensor

        if opt.gpuid >= 0 then
            current_Q = current_Q:float():cuda()
            delta = delta:float():cuda()
        end
        delta:add(-1, current_Q)

        local tem_delta = delta:clone()     -- have to use this cloned tem_delta, since pow() and mul() will directly influence original tensor values.
        loss = loss + tem_delta:pow(2):mul(0.5):cumsum()[obs_train[t]:size(1)] -- this loss here is loss = 0.5 * (y-t)^2

        delta[delta:ge(opt.clip_delta)] = opt.clip_delta
        delta[delta:le(-opt.clip_delta)] = -opt.clip_delta

        dloss_dy[t] = torch.zeros(obs_train[t]:size(1), game_actions:size(1))   -- dim: batch_size * action_num

        if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        dloss_dy[t] = dloss_dy[t]:float():cuda()
        end

        -- Attention: Here the loss calculation is a little different from DQN code. They use the NEGATIVE derivative directly,
        -- and then add that NEGATIVE derivative. I calculate the normal derivative here.
        for i=1, predict_Q_values[t]:size(1) do     -- Here we are trying to get dloss/dy. We still need to dloss from hidden states calculation to be concatenated together for calculating nn.backward()
            dloss_dy[t][i][acts_train[t][i][1]] = delta[i][1] * -1.0    -- dloss_dy equals to the gradient d(loss)/d(y) based on mean squre error. Gradient is -(y-t)
        end
    end

    loss = loss / ((rlTrajLength-1) * obs_train[1]:size(1))   -- loss = loss / ( (trajectory_lenght - 1) * batch_size )

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[rlTrajLength-1] = clone_list(init_state, true)} -- true also zeros the clones. The last drnn_state index should be "rlTrajLength-1", since we'll not backprop error from last time step.
    for t = rlTrajLength-1, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = dloss_dy[t]    -- the return value is d_loss/d_output for a batch at a certain time step
        table.insert(drnn_state[t], doutput_t)  -- drnn_state[t] is a table containing several tensors, with each tensor representing the derivative of loss wrt output to that layer (hidden/cell/output)

        local dlst = clones.rnn[t]:backward( { obs_train[t], unpack(rnn_state[t-1]) }, drnn_state[t] )  -- the 2nd param in module.backward() is dloss/d_output. In rnn, this rnn_output contains real output and hidden state values.
        -- return of this backward() contains dloss/d_input, where input contains both raw input x and previous hidden layer values.
        drnn_state[t-1] = {}
        -- print('*** t:', t, 'dlst size: ', #dlst)
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
            -- note we do k-1 because first item is dembeddings, and then follow the
            -- derivatives of the state, starting at index 2. I know...
            -- It is so determined due to the sequence of how input layers are arranged in this NN.
            drnn_state[t-1][k-1] = v
            -- the entities in dlst corresponds to derivative of loss wrt all input layers. In LSTM, it includes input
            -- layer and various cell state layers and hidden layers from previous time step.
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- clip gradient element-wise
    -- grad_params is calculated when model:backward() is invoked
    -- grad_params contains gradient of error function wrt trainable params.

    grad_params:div(rlTrajLength-1)
    -- Rigth now, grad_param is derivative of loss wrt params.
    -- Try to add L2 norm item here.
    grad_params:add(opt.L2_weight, params)  -- L2 norm is 0.5 * weight * W^2. So the derivative of this item wrt W is (weight * W).
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)    -- clamp() fasten values in the tensor in between this lower and upper bounds.
    return loss, grad_params
end


--- Training
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate }
if opt.gpuid >= 0 then
    obs_train = obs_train:float():cuda()
    acts_train = acts_train:float():cuda()
    rwds_train = rwds_train:float():cuda()
    trms_train = trms_train:float():cuda()
end

--- Fill in the data set
while sample_iter<batchSize or not batch_pos_block_full or not batch_neg_block_full do
    generate_trajectory()   -- Each time only one trajectory was generated
    sample_iter = sample_iter + 1
end
print('Training starts after sample iterations of :', sample_iter)

while sample_iter<=opt.max_epochs do

    generate_trajectory()   -- Each time only one trajectory was generated

    local loss_t
    local timer = torch.Timer()
    for itr=1, opt.train_count do
        _, loss_t = optim.rmsprop(feval, params, optim_state)
    end
    local time = timer:time().real

    if sample_iter % opt.print_freq == 0 then
        local train_loss = loss_t[1][1]
        print(string.format("Iter: %d, rwd: %.1f, grad/param: %.4f, loss: %.6f, time: %d:%d:%d ", sample_iter, rwds_train:sum(),
            grad_params:norm()/params:norm(), train_loss, os.date('*t')['hour'], os.date('*t')['min'], os.date('*t')['sec']))
    end

    -- exponential learning rate decay
    if sample_iter % opt.learning_rate_decay_freq == 0 and opt.learning_rate_decay < 1 then
        if sample_iter >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if opt.target_q > 0 and sample_iter % opt.target_q == 0 then
        set_target_q_network()
        print('Reset target Q function')
    end

    -- every now and then or on last iteration
    if sample_iter % opt.write_every == 0 or sample_iter == opt.max_epochs then
        local savefile = string.format('%s/%d_%s_epoch%d_%.1f.t7', opt.checkpoint_dir, os.time()%1000, opt.savefile, sample_iter, rwds_train:sum())
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.sample_iter = sample_iter
        torch.save(savefile, checkpoint)
    end

    -- do garbage collection
    if sample_iter % opt.gc_freq == 0 then
        io.flush()
        collectgarbage()
    end

    sample_iter = sample_iter + 1
end

print('Episodes: ', episodes, 'Total Reward: ', totalReward, 'Avg: ', totalReward/episodes)
