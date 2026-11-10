# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# The C boundary owns native resources; these files own the MOI model and its
# independently versioned evaluator, native-handle, and completed-result state.
include("utils.jl")
include("MOI_state.jl")
include("MOI_model.jl")
include("MOI_evaluator.jl")
include("MOI_complementarity.jl")
include("MOI_solve.jl")
include("MOI_results.jl")
include("MOI_callbacks.jl")
include("MOI_diagnostics.jl")
