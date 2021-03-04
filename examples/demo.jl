using Revise
using Pkg
Pkg.activate("./")
Pkg.update("NIDDL_FEM")
Pkg.update("NIDDL")
using LinearAlgebra
using SparseArrays
using SuiteSparse
using SharedArrays
using Distributed
using LinearMaps
using IterativeSolvers
using TimerOutputs
using NIDDL_FEM
using NIDDL
using NIDDL_TestCases
using Test
prefix=pwd()*"/data/"

## Mesh and domains
d = 2
k = 1
Nλ = 100
λ(k) = 2π/k
# Timer
to = TimerOutput()
# Geometry
g = LayersGeo(;d=d, shape=[:line,:circle,:sphere,][d], as=[1,],
              interior=true, nΩ=3, mode=:metis, nl=4, layer_from_PBC=true)
# Mesh and domains
h = 2π/abs(k) / max(5, Nλ);
@time m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix*"eraseme");
Ω = union(Ωs...);
Σfull = skeleton(Ω);
Σ = union(Domain.(unique(vcat([[γ for γ in boundary(ω)] for ω in Ωs]...)))...);
# thickened skeleton
ΩΣ = [union([Domain(ω) for ω in Ωj if !isempty(intersect(boundary(Domain(ω)), Σ))]...) for Ωj in Ωs];
save_partition(m, Ωs, prefix*"eraseme");
save_partition(m, [ω for ω in Ωs], prefix*"eraseme_partition");
save_partition(m, [boundary(ω) for ω in Ωs], prefix*"eraseme_skeleton");
save_partition(m, ΩΣ, prefix*"eraseme_thickened_skeleton");
save_partition(m, [Domain(ω) for ω in Ω], prefix*"eraseme_partition_full");
save_partition(m, [Domain(σ) for σ in Σfull], prefix*"eraseme_skeleton_full");
#save_partition(m, [boundary(Domain(γ)) for γ in Σ], prefix*"eraseme_wirebasket");

## Mesh and domains
d = 2
k = 50
Nλ = 20
λ(k) = 2π/k
# Timer
to = TimerOutput()
# Geometry
g = LayersGeo(;d=d, shape=[:line,:circle,:sphere,][d], as=[1,],
              interior=true, nΩ=3, mode=:metis, nl=10, layer_from_PBC=true)
# Mesh and domains
h = 2π/abs(k) / max(5, Nλ);
@time m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix*"eraseme");
Ω = union(Ωs...);
save_partition(m, Ωs, prefix*"eraseme");
save_partition(m, [Domain(ω) for ω in Ω], prefix*"eraseme_partition");
# Medium
μr(x, ielt) = 1
ϵr(x, ielt) = 1
medium = ElectromagneticMedium(;k0=k, μr=μr, ϵr=ϵr, name="")
pb_type = MaxwellPb
ρr(x, ielt) = 1
κr(x, ielt) = 1
medium = AcousticMedium(;k0=k, ρr=ρr, κr=κr, name="")
pb_type = HelmholtzPb
save_medium(m, Ω, medium, prefix*"medium")

## Test case
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[NeumannBC,])
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[NeumannBC,RobinBC,])
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
# Transmission type
tp = Idl2TP(;z=1)
tp = SndOrderTP(;z=1,α=h/(2*k^2))
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
tp = DespresTP(;z=1)
tp = DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
# Solver
solver = Jacobi_S(;tol=1.e-12, maxit=10000, r=0.5, light_mode=false)
solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=true)
# DDM type
dd = OnionDDM(;implicit=true)
dd = JunctionsDDM(;implicit=true, inductive=false, precond=true)
# Problems
fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
# Exact discrete solution
if solver.light_mode
    uexact = zeros(Complex{Float64},0)
else
    @timeit to "Full problem resolution" uexact = solve(m,fullpb);
    # Taking care of potential auxiliary equations
    NΩ = number_of_elements(m,fullpb.Ω,dofdim(fullpb));
    MKtoΩ = Mapping(1:NΩ, 1:NΩ, (NΩ, length(uexact)));
    uexact = MKtoΩ * uexact;
end
# DDM
gid = StandardInputData(m, fullpb, pbs);
ddm = DDM(pbs, gid, dd; to=to);
@timeit to "Resfunc setup" resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact,
    solver; save_solutions_it=false, prefix=prefix, name="eraseme");
@timeit to "Solver" u,x,res = solver(ddm; resfunc=resfunc, to=to);
# Printing some timings
show(to); println("")
# Saving
save_solutions(m, fullpb, solver, u, uexact, prefix, "eraseme");
save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix,
                         "eraseme");