
local OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.eye(outputSize)
end

-- This function nn:updateOutput() computes the output using the current param set of the class and input.
-- This function returns teh result which is stored in the output field.
-- The forward module in the abstract class Module will call updateOutput(input).
function OneHot:updateOutput(input)
  self.output:resize(input:size(1), self.outputSize):zero()     -- As it is used in RNN.lua or LSTM.lua, the input to this function is a 1-dim tensor, # of whose entities is # of data points in one batch.
                                                                -- So, input:size(1) represents the # of data points in one batch. self.outputSize is the vocab_size, which is also the size of the onehot representation dimension.
  if self._eye == nil then self._eye = torch.eye(self.outputSize) end
  self._eye = self._eye:float()
  local longInput = input:long()
  self.output:copy(self._eye:index(1, longInput))   -- the index function here returns a tensor along the 1st dimension, with index of longInput. It actually returns the one-hot representation with the longInput indexed entity activated.
  -- print('## encoding size: ', #self.output, 'End')  -- self.output is a 2-dim tensor, of size (batch_size * vocab_size). So, it returns onehot encodings for batch_size # entities.
  return self.output    -- I've tested. The output is a 2-dim tensor. # of rows is batch_size, # of columns is vocab_size. So, each row is the onehot representation of a specific word in a batch.
end
