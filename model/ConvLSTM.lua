--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 10/31/16
-- Time: 5:06 PM
-- This file is created for using conv net and lstm to solve problems from torch
-- rlenvs, like Catch, in which input features are pixels. The LSTM model implementation
-- is based on Andrej Karpathy's char-rnn program on github. Its integration with Convnet
-- follows the implementation of Deepmind's dqn program.
-- The param convArgs should be a table, in which it contains inputDim (image dimension),
-- outputChannel (table), filterSize (table), filterStride (table), pad (table),
-- applyPooling (boolean).
--

local ConvLSTM = {}
function ConvLSTM.convlstm(output_size, rnn_size, rnn_layer, dropout, convArgs)

    dropout = dropout or 0
    -- there will be 2*n+1 inputs. 1 input, n sets/layers of cell states and hidden states
    local inputs = {}   -- in train.lua, inputs dictionary is designed as the structure containing the input at current time step, hidden state and cell state at previous time step, for lstm.
    table.insert(inputs, nn.Identity()()) -- x. For the ConvLSTM model, the input should be a batch of images(2d or 3d pixels), these inputs should be at the same time step.
    for L = 1, rnn_layer do
--        table.insert(inputs, nn.Identity()()) -- prev_c[L]. The hidden layer data format for ConvLSTM is the same as ordinary lstm.
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    --- Set up convnet structure
    local convInput
    local convOutputs = {}
    local convSingleLayerInChannel, convSingleLayerOutChannel, convSingleLayerInput
    local lastConvLayerWidth = convArgs.inputDim[2]
    local lastConvLayerHeight = convArgs.inputDim[3]
    for conviter = 1, #convArgs.outputChannel do
        if conviter == 1 then
            convSingleLayerInChannel = convArgs.inputDim[1]
            convSingleLayerOutChannel = convArgs.outputChannel[1]
            convSingleLayerInput = inputs[1]    -- inputs[1] is the actual input, which should be a bunch of images.
        else
            convSingleLayerInChannel = convArgs.outputChannel[conviter-1]
            convSingleLayerOutChannel = convArgs.outputChannel[conviter]
            convSingleLayerInput = convOutputs[conviter-1]
        end

        local conv1 = nn.SpatialConvolution(convSingleLayerInChannel, convSingleLayerOutChannel, convArgs.filterSize[conviter], convArgs.filterSize[conviter], convArgs.filterStride[conviter], convArgs.filterStride[conviter])(convSingleLayerInput):annotate{name='conv_'..conviter}  -- For the rlenv problem Catch, input is 1 channel of 24*24 pixels.
        local conv1_nl = nn.ReLU()(conv1):annotate{name='convnl_'..conviter}   -- If using a 2*2 conv window, output of one channel should be of 23*23.
        if convArgs.applyPooling and convArgs.pad[conviter] ~= nil and convArgs.pad[conviter] > 0 then
            convOutputs[conviter] = nn.SpatialMaxPooling(2,2,2,2,1,1)(conv1_nl):annotate{name='convpool_'..conviter }   -- Here we assume Conv layer uses stride of 1.
            lastConvLayerWidth = math.ceil(((lastConvLayerWidth - convArgs.filterSize[conviter]) / convArgs.filterStride[conviter] + 1 + convArgs.pad[conviter]) / 2)
            lastConvLayerHeight = math.ceil(((lastConvLayerHeight - convArgs.filterSize[conviter]) / convArgs.filterStride[conviter] + 1 + convArgs.pad[conviter]) / 2)
        else
            convOutputs[conviter] = conv1_nl
            lastConvLayerWidth = (lastConvLayerWidth - convArgs.filterSize[conviter]) / convArgs.filterStride[conviter] + 1
            lastConvLayerHeight = (lastConvLayerHeight - convArgs.filterSize[conviter]) / convArgs.filterStride[conviter] + 1
        end
    end

    local lastConvLayerNum = convArgs.outputChannel[#convArgs.outputChannel]
    local nel = lastConvLayerNum * lastConvLayerWidth * lastConvLayerHeight    --288 --convOutputs[#convOutputs][1]:nElement()     -- The index [1] indicates one data point in a batch
    local conv_reshape = nn.Reshape(nel)(convOutputs[#convOutputs])

    local x, input_size_L
    local outputs = {}

    -- test
    print('@@@&&&')  print(rnn_layer)
    for L = 1, rnn_layer do
        -- c,h from previos timesteps
        local prev_h = inputs[L*2]  --[L*2+1]
--        local prev_c = inputs[L*2]
        -- the input to this layer
        if L == 1 then
            x = conv_reshape    -- The reshaped tensor of convnet's output
            input_size_L = nel -- This is the number of features of one data point after convolution/max pooling
        else
            x = outputs[(L-1)*2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end
        -- evaluate the input sums at once for efficiency
        local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})   -- all_input_sums is of size (batch_size * (4*rnn_size)), it's 2d.

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)  -- Got it. nn.Reshape(4, rnn_size) reorganize all data entities in the 2-d tensor into a 3-d tensor.
        --                                                        -- The 2nd and 3rd dim are of size 4 and rnn_size respectively. The 1st dim's size of the original entity
        --                                                        -- quantity divided by 4*rnn_size. The organizing of data in new 3-d tensor is along the row priority of
        --                                                        -- the previous 2-d tensor. So, all original data were cut into 4*rnn_size blocks, along row direction.
        table.insert(outputs, reshaped)

--        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)  -- nn.SplitTable(dim) returns a module which could split a tensor along the dim dimension.
--        -- Then the __call__ func takes the reshaped param. My understanding is like the __call__
--        -- just likes performs the forward func.
--        -- This split(4) func is nnNode:split(). It returns param # of nodes, each takes a single component of the output. https://github.com/torch/nngraph/blob/c654b19a11004ffa1061ca689f1ca89fe527cbe7/node.lua#L43-L59
--        -- decode the gates
--        local in_gate = nn.Sigmoid()(n1)
--        local forget_gate = nn.Sigmoid()(n2)
--        local out_gate = nn.Sigmoid()(n3)
--        -- decode the write inputs
--        local in_transform = nn.Tanh()(n4)
--        -- perform the LSTM update
--        local next_c = nn.CAddTable()({
--            nn.CMulTable()({forget_gate, prev_c}),
--            nn.CMulTable()({in_gate,     in_transform})
--        })
--        -- gated cells form the output
--        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
--
--        table.insert(outputs, next_c)  -- pay a little attention to the sequence of entities in outputs table. Each time step, 2 things will be stored in outputs.
--        table.insert(outputs, next_h)  -- The 1st if candidate state, the 2nd is outputted hidden state. It influences how prev_c and prev_h are accessed.
    end

--    -- set up the decoder
--    local top_h = outputs[#outputs]
--    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
--    local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
--    --  local logsoft = nn.LogSoftMax()(proj)
--    table.insert(outputs, proj)    -- graph.dot(nn.gModule(inputs, outputs).fg, 'MLP', 'outputBasename') -- This graph.dot() could be used to draw the nngraph graph.

    return nn.gModule(inputs, outputs)
end

return ConvLSTM



