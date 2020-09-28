function detect_junctions(m::Mesh, Ωs::Vector{Domain}, d::Int64)
    # Initialisation
    eltInNdomains = zeros(UInt16, number_of_elements(m,d))
    # Loop on domains
    for ω in Ωs
        eltInNdomains[element_indices(m,ω,d)] .+= 1
    end
    # Junctions
    bool_j = eltInNdomains .>= 2
    indices = (1:number_of_elements(m,d))[bool_j] # DOF indices
    weights = eltInNdomains[bool_j]
    return indices, weights
end


function junction_weights(m::Mesh, Ωs::Vector{Domain}, pb)
    # Detection of junction points
    ind, weights = detect_junctions(m, Ωs, dofdim(pb))
    # Initialisation
    N = number_of_elements(m, union(Ωs...), dofdim(pb))
    # Treatment of junction points
    Cws = sparse(ind, ind, 1 ./ weights, N, N)
    return Cws
end


struct InputData <: AbstractInputData
    m::Mesh
    RΩ::SparseMatrixCSC{Bool,Int64}
    RΣ::SparseMatrixCSC{Bool,Int64}
    Cwsst::SparseMatrixCSC{Float64,Int64}
    NΣis::Vector{Int64}
end
function InputData(m::Mesh, fullpb::P, pbs::Vector{P};
                   to=missing) where P <: AbstractProblem
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @timeit to "Global input data" begin
        @info "   --> #DOF volume   $(number_of_elements(m,fullpb.Ω,dofdim(fullpb)))"
        @info "   --> #DOF skeleton $(number_of_elements(m,skeleton(fullpb.Ω),dofdim(fullpb)))"
        # Skeleton
        Σ = union(Domain.(unique(vcat([transmission_boundary(pb)[:] for pb in pbs]...)))...)
        # Definition of some restriction matrices
        @timeit to "RΩ" RΩ = restriction(m,fullpb.Ω,dofdim(fullpb))
        @timeit to "RΣ" RΣ = restriction(m,Σ,dofdim(fullpb))
        # Diagonal matrix of junction weights at transmission DOFs
        @timeit to "Junction weights" Cws = junction_weights(m, [p.Ω for p in pbs], fullpb)
        @timeit to "Cwsst" Cwsst = RΣ * Cws * transpose(RΣ)
        # Size
        @timeit to "NΣis" NΣis = [number_of_elements(m,transmission_boundary(pb),dofdim(pb)) for pb in pbs]
    end
    InputData(m, RΩ, RΣ, Cwsst, NΣis)
end
