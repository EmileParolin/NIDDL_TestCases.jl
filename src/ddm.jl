struct InputData <: AbstractInputData
    m::Mesh
    d::Integer                    # Dimension of DOF ∈ {0,1}
    Ω::Domain                     # Full domain
    Σ::Domain                     # Full skeleton
    NΩis::Vector{Int64}           # Size of local problems
    NΣis::Vector{Int64}           # Size of local transmission boundaries
    dof_weights::Vector{Int64}    # Number of domains a DOF belongs to
end
function InputData(m::Mesh, fullpb::P,
                   pbs::Vector{P}) where P <: AbstractProblem
    # Dimension of DOFs
    d = dofdim(fullpb)
    # Domains
    Ω = fullpb.Ω
    Σ = union(Domain.(unique(vcat([transmission_boundary(pb)[:] for pb in pbs]...)))...)
    Ωs = [pb.Ω for pb in pbs] 
    Σs = [transmission_boundary(pb) for pb in pbs] 
    # Size
    @info "   --> #DOF volume   $(number_of_elements(m,Ω,d))"
    @info "   --> #DOF skeleton $(number_of_elements(m,Σ,d))"
    NΩis = [number_of_elements(m,ω,d) for ω in Ωs]
    NΣis = [number_of_elements(m,σ,d) for σ in Σs]
    # Number of domains a DOF belongs to
    dof_weights = zeros(Int64, number_of_elements(m,d))
    for ω in Ωs
        dof_weights[element_indices(m,ω,d)] .+= 1
    end
    InputData(m, d, Ω, Σ, NΩis, NΣis, dof_weights)
end


function indices_full_domain(gid::InputData)
    element_indices(gid.m, gid.Ω, gid.d)
end


function indices_skeleton(gid::InputData)
    element_indices(gid.m, gid.Σ, gid.d)
end


function indices_domain(gid::InputData, pb::Problem)
    element_indices(gid.m, pb.Ω, dofdim(pb))
end


function indices_transmission_boundary(gid::InputData, pb::Problem)
    element_indices(gid.m, transmission_boundary(pb), dofdim(pb))
end


size_multi_trace(gid::InputData) = gid.NΣis
dof_weights(gid::InputData) = gid.dof_weights
get_matrix(gid::InputData, pb::Problem) = get_matrix(gid.m, pb)
get_matrix_no_transmission_BC(gid::InputData, pb::Problem) = get_matrix_no_transmission_BC(gid.m, pb)
get_rhs(gid::InputData, pb::Problem) = get_rhs(gid.m, pb)

function get_transmission_matrix(gid::InputData, pb::Problem)
    RΣi = restriction(gid.m, transmission_boundary(pb), dofdim(pb))
    # Computation of transmission matrix
    Ti = spzeros(Float64, size(RΣi, 1), size(RΣi, 1))
    # Loop needed in case of more than one connected transmission boundary
    for bc in pb.BCs
        if typeof(bc) <: TransmissionBC
            RΣii = restriction(gid.m, bc.Γ, dofdim(pb))
            Ti += RΣi * transpose(RΣii) * matrix(gid.m,pb,bc) * RΣii * transpose(RΣi)
        end
    end
    return Ti
end


function DtN(gid::InputData, pb::Problem)
    Σ = remove(union([bc.Γ for bc in pb.BCs if typeof(bc) <: PhysicalBC]...),
               boundary(pb.Ω))
    Λi = pb.medium.k0 * DtN(gid.m, pb, Σ)
end
