"""
In the electromagnetic setting, we have, variationnally

    (K - ik T) E = - J

We define Λ as the DtN

    Λ : H^{-1/2}_curl → H^{-1/2}_div
        γ0 E = n± × (E × n±) ↦ γ1 = (1/ik) n± × curl E

"""
function DtN(m::Mesh, pb::Problem, Σ::Domain)
    # Removing transmission conditions
    bcs = filter(bc->typeof(bc) <: PhysicalBC, pb.BCs)
    # Problem matrix
    fake_pb = typeof(pb)(pb.medium, pb.Ω, bcs)
    Ktilde = get_matrix(m,fake_pb) # TBC not applied in get_matrix
    @assert size(Ktilde,1) == size(Ktilde,2)
    # Definition of some sizes
    NΣ = number_of_elements(m,Σ,dofdim(pb))
    NK = size(Ktilde,1)
    N0 = NK - NΣ
    # Definition of some restriction matrices
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    RΣ = restriction(m,Σ,dofdim(pb))
    IK = sparse(I,NK,NK)
    IΣinK = sparse(findnz(RΩ*transpose(RΣ)*RΣ*transpose(RΩ))...,NK,NK)
    # Mappings
    i,j,v = findnz(IΣinK)
    MKtoΣ = sparse(collect(1:length(j)),j,ones(Bool,length(j)),length(j),NK)
    i,j,v = findnz(IK - IΣinK)
    MKto0 = sparse(collect(1:length(j)),j,ones(Bool,length(j)),length(j),NK)
    # 2x2 block matrix
    K00 = MKto0 * Ktilde * transpose(MKto0)
    KΣΣ = MKtoΣ * Ktilde * transpose(MKtoΣ)
    K0Σ = MKto0 * Ktilde * transpose(MKtoΣ)
    KΣ0 = MKtoΣ * Ktilde * transpose(MKto0)
    # DtN via Schur complement
    Λ = -(1/(im*pb.medium.k0))*(KΣΣ - KΣ0 * inv(Matrix(K00)) * K0Σ)
    return Λ
end
