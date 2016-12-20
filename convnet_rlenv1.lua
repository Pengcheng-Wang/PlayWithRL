--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 12/20/16
-- Time: 5:12 PM
-- To change this template use File | Settings | File Templates.
--

return function(convArgs)
    convArgs.outputChannel = {12, 24}    -- could have multiple layers
    convArgs.filterSize = {4, 2}
    convArgs.filterStride = {2, 1}
    convArgs.pad = {1}
    convArgs.applyPooling = {false}
end