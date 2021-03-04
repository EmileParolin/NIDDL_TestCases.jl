using Revise, Pkg, Test
Pkg.activate("./")
using LinearAlgebra, SparseArrays, SuiteSparse, SharedArrays, Distributed
using LinearMaps, IterativeSolvers, TimerOutputs
using NIDDL_FEM, NIDDL, NIDDL_TestCases
using Colors, PGFPlotsX
using WriteVTK
prefix = pwd() * "/data/"


## Helper function
function run(d, k, Nλ, as, nΩ, νs, name; nl=0, fbc=:robin)
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                  interior=true, nΩ=nΩ, mode=:metis, nl=nl, layer_from_PBC=true)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix * name);
    Ω = union(Ωs...);
    Σ = union(Domain.(unique(vcat([[γ for γ in boundary(ω)] for ω in Ωs]...)))...);
    # thickened skeleton
    ΩΣ = [union([Domain(ω) for ω in Ωj if !isempty(intersect(boundary(Domain(ω)), Σ))]...) for Ωj in Ωs];
    save_partition(m, [ω for ω in Ωs], prefix*"eraseme_partition");
    save_partition(m, [boundary(ω) for ω in Ωs], prefix*"eraseme_skeleton");
    save_partition(m, ΩΣ, prefix*"eraseme_thickened_skeleton");
    if d == 3
        save_partition(m, [boundary(Domain(γ)) for γ in Σ], prefix*"eraseme_wirebasket");
    end
    # Test case
    medium = AcousticMedium(;k0=k)
    pb_type = HelmholtzPb
    tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
    # DDM type
    dd = JunctionsDDM(;implicit=false, precond=true)
    ## Reference
    # Problems
    tp = Idl2TP(;z=1)
    tp = DespresTP(;z=1)
    fullpb_ref, pbs_ref = get_problems(g, tc, Ωs, Γs, tp, dd);
    # DDM
    gid_ref = StandardInputData(m, fullpb_ref, pbs_ref);
    ddm_ref = DDM(pbs_ref, gid_ref, dd);
    ## Exchange operator matrix
    Π_ref = vcat([ld.Πi.Πi for ld in ddm_ref.lds]...);
    ## Transmission operator
    Tis_ref = [ld.Πi.Ti for ld in ddm_ref.lds]
    ## Full transmission operator
    Tis = [ld.Πi.Ti for ld in ddm_ref.lds]
    PΣitoΣmts = [matrix(ld.Πi.MΣitoΣmt) for ld in ddm_ref.lds]
    T0MT_MTt = sum([PΣitoΣmt * Ti * transpose(PΣitoΣmt)
                    for (PΣitoΣmt, Ti) in zip(PΣitoΣmts, Tis)])
    TM_ref = Matrix(T0MT_MTt)
    ## Operator behind projection
    Tis = [ld.Πi.Ti for ld in ddm_ref.lds]
    PΣitoΣsts = [matrix(ld.Πi.MΣitoΣst) for ld in ddm_ref.lds]
    T0ST_STt = sum([PΣitoΣst * Matrix(Ti) * transpose(PΣitoΣst)
                    for (PΣitoΣst, Ti) in zip(PΣitoΣsts, Tis)])
    TS_ref = Matrix(T0ST_STt)
    ## Weights (used in preconditioner)
    ind_Ω = indices_full_domain(gid_ref)
    ind_Σ = indices_skeleton(gid_ref)
    MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
    weights = MΩtoΣ * (1 ./ dof_weights(gid_ref))
    ## Preconditioner
    PT0ST_STt = sum([PΣitoΣst * inv(Matrix(Ti)) * transpose(PΣitoΣst)
                    for (PΣitoΣst, Ti) in zip(PΣitoΣsts, Tis)])
    PT_ref = diagm(0=>weights[:]) * PT0ST_STt * diagm(0=>weights[:])
    ## Projector
    Trhs = sum([PΣitoΣst * Matrix(Ti) * transpose(PΣitoΣmt)
                for (PΣitoΣmt, PΣitoΣst, Ti) in zip(PΣitoΣmts, PΣitoΣsts, Tis)])
    P_ref = inv(TS_ref) * Matrix(Trhs)
    ## Projector (with reinterpretation as MT)
    Pfull_ref = sum([PΣitoΣmt * transpose(PΣitoΣst) * P_ref
                for (PΣitoΣmt, PΣitoΣst) in zip(PΣitoΣmts, PΣitoΣsts)])
    ## Ramp in dissipation parameter
    Tiss = Vector{Matrix{Complex{Float64}}}[]
    Kss = Vector{Vector{SparseMatrixCSC}}[]
    TMs = Matrix{Complex{Float64}}[]
    TSs = Matrix{Complex{Float64}}[]
    PTs = Matrix{Complex{Float64}}[]
    Ps = Matrix{Complex{Float64}}[]
    Pfulls = Matrix{Complex{Float64}}[]
    Πs = Matrix{Complex{Float64}}[]
    for ν in νs
        # Problems
        medium = AcousticMedium(;k0=ν)
        diss_medium = dissipative_medium(medium)
        tp = DtN_TP(;z=(ν < k ? ν/k : 1),pb_type=pb_type,medium=diss_medium,fbc=fbc)
        fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
        # DDM
        gid = StandardInputData(m, fullpb, pbs);
        ddm = DDM(pbs, gid, dd);
        ## Exchange operator matrix
        Πmat = vcat([ld.Πi.Πi for ld in ddm.lds]...);
        push!(Πs, Πmat);
        ## Transmission operator
        Tis = [ld.Πi.Ti for ld in ddm.lds]
        push!(Tiss, Tis);
        ## Full transmission operator
        Tis = [ld.Πi.Ti for ld in ddm.lds]
        PΣitoΣmts = [matrix(ld.Πi.MΣitoΣmt) for ld in ddm.lds]
        T0MT_MTt = sum([PΣitoΣmt * Ti * transpose(PΣitoΣmt)
                        for (PΣitoΣmt, Ti) in zip(PΣitoΣmts, Tis)])
        TM = Matrix(T0MT_MTt)
        push!(TMs, TM);
        ## Operator behind projection
        Tis = [ld.Πi.Ti for ld in ddm.lds]
        PΣitoΣsts = [matrix(ld.Πi.MΣitoΣst) for ld in ddm.lds]
        T0ST_STt = sum([PΣitoΣst * Matrix(Ti) * transpose(PΣitoΣst)
                        for (PΣitoΣst, Ti) in zip(PΣitoΣsts, Tis)])
        TS = Matrix(T0ST_STt)
        push!(TSs, TS);
        ## Weights (used in preconditioner)
        ind_Ω = indices_full_domain(gid)
        ind_Σ = indices_skeleton(gid)
        MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
        weights = MΩtoΣ * (1 ./ dof_weights(gid))
        ## Preconditioner
        PT0ST_STt = sum([PΣitoΣst * inv(Matrix(Ti)) * transpose(PΣitoΣst)
                        for (PΣitoΣst, Ti) in zip(PΣitoΣsts, Tis)])
        PT = diagm(0=>weights[:]) * PT0ST_STt * diagm(0=>weights[:])
        push!(PTs, PT);
        ## Projector
        Trhs = sum([PΣitoΣst * Matrix(Ti) * transpose(PΣitoΣmt)
                    for (PΣitoΣmt, PΣitoΣst, Ti) in zip(PΣitoΣmts, PΣitoΣsts, Tis)])
        P = inv(TS) * Matrix(Trhs)
        push!(Ps, P);
        ## Projector (with reinterpretation as MT)
        Pfull = sum([PΣitoΣmt * transpose(PΣitoΣst) * P
                    for (PΣitoΣmt, PΣitoΣst) in zip(PΣitoΣmts, PΣitoΣsts)])
        push!(Pfulls, Pfull);
    end
    info_ref = Tis_ref, Π_ref, TM_ref, TS_ref, PT_ref, P_ref, Pfull_ref
    infos = Tiss, Πs, TMs, TSs, PTs, Ps, Pfulls
    return m, Ω, Ωs, gid_ref, ddm_ref, info_ref, infos
end

## General parameters
d = 2
k = 10
νs = k .* 2. .^ (0:1:0)
νs = k .* 2. .^ (-3.5:0.5:4.0)
Nλ = 40
as = [1,]
nΩ = 3
name = "eraseme"
info = run(d, k, Nλ, as, nΩ, νs, name; nl=0, fbc=:robin);
m, Ω, Ωs, gid_ref, ddm_ref, info_ref, infos = info;
Tis_ref, Π_ref, TM_ref, TS_ref, PT_ref, P_ref, Pfull_ref = info_ref;
Tiss, Πs, TMs, TSs, PTs, Ps, Pfulls = infos;

## Vizualization
# In Paraview:
# - set direction to -z
# - rotate 90deg clockwise
for i in 1:length(νs)
    # Sanity check for orientation if necessary
    M = zeros(5,5)
    M[1,1:end] = collect(1:size(M, 2))
    M[end,1:end] = collect(size(M, 2):-1:1)
    # Sanity checks on imaginary parts
    @assert norm(abs.(imag.(Tiss[i][1]))) < 1.e-12
    @assert norm(abs.(imag.(TMs[i]))) < 1.e-12
    @assert norm(abs.(imag.(TSs[i]))) < 1.e-12
    @assert norm(abs.(imag.(Ps[i]))) < 1.e-12
    # Transmission operator
    M = abs.(real.(Tiss[i][1]))
    # Exchange operator
    M = abs.(real.(Πs[i]))
    M = abs.(real.(Π_ref))
    # Scalar product matrix
    M = abs.(real.(TMs[i]))
    M = abs.(real.(TM_ref))
    # Full projector
    M = abs.(real.(Pfulls[i]))
    M = abs.(real.(Pfull_ref))
    # Projector
    M = abs.(real.(Ps[i]))
    M = abs.(real.(P_ref))
    # System
    M = abs.(real.(TSs[i]))
    M = abs.(real.(TS_ref))
    # Inverse of system
    M = abs.(real.(inv(TSs[i])))
    M = abs.(real.(inv(TS_ref)))
    # Preconditioned system
    M = abs.(real.(PTs[i] * TSs[i]))
    M = abs.(real.(PT_ref * TS_ref))
    vtk_write_array(prefix*"eraseme_$(i)", M, "value")
end

## Visualization exchange
# see vizu_exchange.jl
# Size of simple trace and multi trace
Nst = number_of_elements(m, skeleton(Ω), 0)
Nmt = sum([number_of_elements(m, boundary(ω), 0) for ω in Ωs])
# weights
ixpt = argmax(dof_weights(gid_ref))
# Indices of domain `iΩ` boundary
iΩ = 1
inds = inds_boundaries(nΩ, ddm_ref, Nst, Nmt)
f_xpt = i -> prod([inds[i][j] > 0 for j in 1:nΩ])
f_tra = i -> inds[i][iΩ] > 0 && (sum([inds[i][j] > 0 for j in 1:nΩ]) == 2)
f_ext = i -> inds[i][iΩ] > 0 && (sum([inds[i][j] > 0 for j in 1:nΩ]) == 1)
inds_xpt = [inds[i][iΩ] for i in 1:Nst if f_xpt(i)]
inds_tra = [inds[i][iΩ] for i in 1:Nst if f_tra(i)]
inds_ext = [inds[i][iΩ] for i in 1:Nst if f_ext(i)]

## Error: no real convergence for a random input
# input
xmt = randn(Nmt);
# output
err = Float64[]
ymt_ref = real.(Π_ref*xmt);
for Π in Πs
    ymt = real.(Π*xmt);
    push!(err, norm(ymt .- ymt_ref))
end
err
# 

## Vizu (export Paraview)
# input
ip_source = inds_ext[1]  # exterior boundary
ip_source = inds_xpt[1]  # interior cross point
ip_source = inds_tra[1]  # boundary cross point
ip_source = inds_tra[30] # transmission cross point 2/3 2/3 | 4 11 13 15 17 19 30 50 104 108
ip_source = inds_tra[10] # transmission cross point 1/2 1/2 | 5 6 7 8 9 10 12 14 16 40 105 107 109
ip_source = inds_tra[60] # transmission cross point 1/3 1/3 | 60 106 110
ip_source = inds_tra[116]
xmt = zeros(Nmt);
xmt[ip_source] = 1.;
xmts = [transpose(ld.MΣitoΣmt) * xmt for ld in ddm_ref.lds]
vtk_save_on_skeleton(m, Ωs, xmts, "input", prefix*"exchange_input");
# projection output
ymt = real.(Pfull_ref*xmt);
println(sort(abs.(ymt))[end-4:end-0])
ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
vtk_save_on_skeleton(m, Ωs, ymts, "output", prefix*"projection_output_0");
for (i, Pfull) in enumerate(Pfulls)
    ymt = real.(Pfull*xmt);
    println(sort(abs.(ymt))[end-4:end-0])
    ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
    vtk_save_on_skeleton(m, Ωs, ymts, "output", prefix*"projection_output_$(i)");
end
println("---------------------------------------------")
# exchange output
ymt = real.(Π_ref*xmt);
#println(sort(abs.(ymt))[end-4:end-0])
ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
vtk_save_on_skeleton(m, Ωs, ymts, "output", prefix*"exchange_output_0");
for (i, Π) in enumerate(Πs)
    ymt = real.(Π*xmt);
    #println(sort(abs.(ymt))[end-4:end-0])
    ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
    vtk_save_on_skeleton(m, Ωs, ymts, "output", prefix*"exchange_output_$(i)");
end
println("---------------------------------------------")

## PROJECTION (J=3)
# - exterior boundary
# With the mass matrix  : = 1
# With the non local op : → 1
# - transmission point
# With the mass matrix  : = 1/2 1/2
# With the non local op : → 2/3 2/3 or
#                           1/2 1/2 or
#                           1/3 1/3     !!!
# - boundary junction point
# With the mass matrix  : = 1/2 1/2 (+ ≂̸ 0)
# With the non local op : → ?   ?
# - interior junction point
# With the mass matrix  : = 1/3 1/3 1/3 (+ ≂̸ 0)
# With the non local op : → 1/6 1/6 1/6