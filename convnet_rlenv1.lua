--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 12/20/16
-- Time: 5:12 PM
-- To change this template use File | Settings | File Templates.
--

return function(convArgs)
    convArgs.outputChannel = {12, 24, 24}    -- could have multiple layers
    convArgs.filterSize = {6, 3, 2}     -- In Catch, input are of size 1*24*24.
    convArgs.filterStride = {3, 2, 1}   -- 12*7*7, 24*3*3, 24*2*2
    convArgs.pad = {1}
    convArgs.applyPooling = {false}     -- No pooling
end