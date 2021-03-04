using Revise, Pkg, Test
Pkg.activate("./")
using LinearAlgebra, SparseArrays, SuiteSparse, SharedArrays, Distributed
using LinearMaps, IterativeSolvers, TimerOutputs
using NIDDL_FEM, NIDDL, NIDDL_TestCases
using Colors, PGFPlotsX
using WriteVTK
prefix = pwd() * "/data/"


## Helper function
function run(k, Nλ, as, nΩ, νs, name)
    d = 2
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                  interior=true, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix * name);
    Ω = union(Ωs...);
    save_partition(m, [Domain(ω) for ω in Ω], prefix * name * "_partition");
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
    gid_ref = InputData(m, fullpb_ref, pbs_ref);
    ddm_ref = DDM(pbs_ref, gid_ref, dd);
    ## Ramp in dissipation parameter
    Tiss = Vector{Matrix{Float64}}[]
    Tis = [ld.Πi.Ti for ld in ddm_ref.lds]
    push!(Tiss, Tis);
    for ν in νs
        # Problems
        diss_medium = dissipative_medium(AcousticMedium(;k0=ν))
        tp = DtN_TP(;z=1,pb_type=pb_type,medium=diss_medium,fbc=:robin)
        fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
        # DDM
        gid = InputData(m, fullpb, pbs);
        ddm = DDM(pbs, gid, dd);
        ## Operator
        Tis = [real.(ld.Πi.Ti) for ld in ddm.lds]
        push!(Tiss, Tis);
    end
    return m, Ω, Ωs, gid_ref, ddm_ref, Tiss
end

## General parameters
k = 1
νs = k .* 2. .^ (-2:0.5:5.5)
Nλ = 100
as = [1,]
nΩ = 1
name = "eraseme"
m, Ω, Ωs, gid_ref, ddm_ref, Tiss = run(k, Nλ, as, nΩ, νs, name);

## Vizualization
# In Paraview:
# - set direction to -z
# - rotate 90deg clockwise
for i in 1:length(νs)
    # Sanity check for orientation if necessary
    M = zeros(5,5)
    M[1,1:end] = collect(1:size(M, 2))
    M[end,1:end] = collect(size(M, 2):-1:1)
    # Transmission operator
    M = abs.(Tiss[i][1])
    vtk_write_array(prefix*"eraseme_$(i)", M, "value")
end

## Beautiful Vizualization (but too expensive)
#p = pgfplotsx_spy(M; separators=Int64[], tol=1.e-12);
#pgfplot_save(prefix*"eraseme.pdf", p)