--
-- User: pwang8
-- Date: 12/26/16
-- Time: 9:08 PM
-- This script offers an implementation of LSTM for regression.
-- The main difference from it in LSTM.lua implementation is that,
-- this script should support configurable multi-layer LSTM.
-- We assume the one input is a one-dimension tensor
--

local LstmReg = {}
function LstmReg.lstm(input_size, output_size, rnn_size_conf, dropout)
    --- Param meaning:
    -- rnn_size_conf - should be a table containing number of neurons in each rnn layer

    local rnn_layers = #rnn_size_conf
    assert(rnn_layers > 0, 'Invalid rnn_size_conf')
    dropout = dropout or 0

    -- there will be 2*rnn_layers+1 inputs. 1 input, rnn_layers sets/layers of cell states and hidden states
    local inputs = {}   -- in train.lua, inputs dictionary is designed as the structure of containing the input at current time step, hidden state and cell state at previous time step, for lstm.
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1,rnn_layers do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    for L = 1,rnn_layers do
        -- c,h from previos timesteps
        local prev_c = inputs[L*2]
        local prev_h = inputs[L*2+1]
        -- the input to this layer
        if L == 1 then
            x = inputs[1]
            input_size_L = input_size
        else
            x = outputs[(L-1)*2]    -- outputs contains prev_c and prev_h from previous layers
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size_conf[L-1]
        end
        -- evaluate the input sums at once for efficiency
        local i2h = nn.Linear(input_size_L, 4 * rnn_size_conf[L])(x):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size_conf[L], 4 * rnn_size_conf[L])(prev_h):annotate{name='h2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})   -- all_input_sums is of size (batch_size * (4*rnn_size_conf[L])), it's 2d.

        local reshaped = nn.Reshape(4, rnn_size_conf[L])(all_input_sums)  -- Got it. nn.Reshape(4, rnn_size) reorganize all data entities in the 2-d tensor into a 3-d tensor.
        --                                                        -- The 2nd and 3rd dim are of size 4 and rnn_size respectively. The 1st dim's size of the original entity
        --                                                        -- quantity divided by 4*rnn_size. The organizing of data in new 3-d tensor is along the row priority of
        --                                                        -- the previous 2-d tensor. So, all original data were cut into 4*rnn_size blocks, along row direction.
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)  -- nn.SplitTable(dim) returns a module which could split a tensor along the dim dimension.
        -- Then the __call__ func takes the reshaped param. My understanding is like the __call__
        -- just likes performs the forward func.
        -- This split(4) func is nnNode:split(). It returns param # of nodes, each takes a single component of the output. https://github.com/torch/nngraph/blob/c654b19a11004ffa1061ca689f1ca89fe527cbe7/node.lua#L43-L59
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate,     in_transform})
        })
        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

        table.insert(outputs, next_c)  -- pay a little attention to the sequence of entities in outputs table. Each time step, 2 things will be stored in outputs.
        table.insert(outputs, next_h)  -- The 1st if candidate state, the 2nd is outputted hidden state. It influences how prev_c and prev_h are accessed.
    end

    -- set up the decoder
    local top_h = outputs[#outputs]
    if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
    local proj = nn.Linear(rnn_size_conf[#rnn_size_conf], output_size)(top_h):annotate{name='predictor'}
    --  local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, proj)    -- graph.dot(nn.gModule(inputs, outputs).fg, 'MLP', 'outputBasename') -- This graph.dot() could be used to draw the nngraph graph.

    return nn.gModule(inputs, outputs)
end

return LstmReg

