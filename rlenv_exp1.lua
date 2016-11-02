require 'torch'
require 'nn'
require 'nngraph'

local image = require 'image'
local Catch = require 'rlenvs/Catch'
local ConvLSTM = require 'model.ConvLSTM'

-- Detect QT for image display
local qt = pcall(require, 'qt')

-- Initialise and start environment
local env = Catch({level = 2})
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()
local observation = env:start()

local reward, terminal
local episodes, totalReward = 0, 0
local batchSize = 50
local nSteps = batchSize * (stateSpec[2][2] - 1)    -- stateSpec[2][2] is the size of the ball falling space. Minus 1 means the distance the ball would move down

cmd = torch.CmdLine()
cmd:option('-rnn_size', 8, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-learning_rate',2e-5,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
convArgs = {}
convArgs.inputDim = stateSpec[2]     -- input image dimension
convArgs.outputChannel = {2}
convArgs.filterSize = {2}
convArgs.filterStride = {1}
convArgs.pad = {0}

-- Display
local window = qt and image.display({image=observation, zoom=10})

-- ConvLSTM model
protos = {}     -- In the char-rnn program, protos is a table contains 2 entities, the 'rnn' entity and the 'criterion' entity. I guess the torch criterion might not be needed in this rl training.
protos.convrnn = ConvLSTM.convlstm((actionSpec[3][2] - actionSpec[3][1]), 8, 2, 0.2, convArgs)

for i = 1, nSteps do
    -- Pick random action and execute it
    local action = torch.random(actionSpec[3][1], actionSpec[3][2])
    reward, observation, terminal = env:step(action)
    totalReward = totalReward + reward

    -- Display
    if qt then
        image.display({image=observation, zoom=10, win=window})
    end


    -- If game finished, start again
    if terminal then
        episodes = episodes + 1
        observation = env:start()
    end
end
print('Episodes: ' .. episodes)
print('Total Reward: ' .. totalReward)