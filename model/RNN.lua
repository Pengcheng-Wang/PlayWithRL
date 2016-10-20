local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout)    -- rnn_size is the number of hidden neurons in one rnn hidden layer. n is the number of hidden layers. input_size is vocabulary size.

  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x  -- nn.Identity() returns a module. A call function () after it just calls the __call__ of the returned module of nn.Identity()
                                        -- It is said by default this __call__ method perfoms a forward/backward, but here this __call__ is overrided. http://stackoverflow.com/questions/30983354/is-there-special-meaning-for-syntax-in-lua
                                        -- The consequence of nn.Identity()() has here the effect of returning a nngraph.Node({module=self}) node where self refers to the current nn.Identity() instance.
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
                                          -- nn.Identity()() returns a nngraph node. It is a graph node, which might mean it also needs tensor input, and output the same entity as input.
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    if L == 1 then
      x = OneHot(input_size)(inputs[1])     -- after OneHot()(), x becomes a 2-dim tensor, each row of which represent a OneHot representation of a sigle character, and # of rows equal to batch size.
      input_size_L = input_size
    else
      x = outputs[(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)    -- the nn.gModule() returns a module with standard API of forward() and backward()
end

return RNN
