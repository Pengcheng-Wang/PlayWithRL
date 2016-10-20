
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is
-- why it is kind of a large

require 'torch'
local model_utils = {}
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}   -- The parameters and gradParameters defined in an rnn model are the trainable parameters and their gradients of the energy wrt
    local gradParameters = {}   -- the learnable parameters. The parameters() func is defined in module.
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()  -- the parameters() function returns 2 tables. First is a table of learnable params {weights}, second is a table of gradients of energy wrt learnable params {gradWeights}.

        if net_params then
            for k, p in pairs(net_params) do
                parameters[#parameters + 1] = p
                -- print(string.format('Param %s has dim: %s, size %s\n', k, p:dim(), p:size()))   -- RNN, LSTM, GRU appears to have pretty different param settings. RNN and LSTM have similar param structures, but different # of params. GRU has pretty different param structure.
            end
            for k, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
                -- print(string.format('Param Gradient %s has dim: %s, size %s\n', k, g:dim(), g:size()))
            end
            -- print(string.format("Let's see dimension of table parameters is: %s\n", table.getn(parameters)))
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]  -- torch.pointer() returns a unique id (pointer) of the given object.
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)  -- unpack() returns all elements from the array in input. In this specific usage, offset gets the 2nd element in the array, and other elements are discarded.
        return offset   -- unpack() is a function in lua which unfold an array.
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}     -- It seems like the storages table keeps pairs of parameter group pointers and their corresponding relative starting position wrt their size.
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()    -- For explanation of Tensor.storage() function (actually, it returns the Storage of all elements of that tensor), refer to https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.storage

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)    -- cumsum(x, n) returns the cumulative sum of elements of x, performing the operation over dimension n.
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])   -- Right now, I'm still not clear if this clone_many_times() func does deep copy or shallow. The set() func seems like doing shallow copy.
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

return model_utils
