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

d = 2
k = 5
Nλ = 30
λ(k) = 2π/k
# Timer
to = TimerOutput()
# Geometry
g = LayersGeo(;d=d, shape=[:circle,:sphere,][d-1], as=[1,2],
              interior=false, nΩ=16, mode=:metis, nl=3, layer_from_PBC=true)
# Mesh and domains
h = 2π/abs(k) / max(5, Nλ);
@time m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix*"eraseme");
Ω = union(Ωs...);
save_partition(m, Ωs, prefix*"eraseme");
save_partition(m, [Domain(ω) for ω in Ω], prefix*"eraseme_partition");
# Medium
μr(x) = 1
ϵr(x) = 1
medium = ElectromagneticMedium(;k0=k, μr=μr, ϵr=ϵr, name="")
ρr(x) = 1
κr(x) = 1
medium = AcousticMedium(;k0=k, ρr=ρr, κr=κr, name="")
save_medium(m, Ω, medium, prefix*"medium")

# Test case
tc = RandomTC(;d=d, medium=medium)
tc = ScatteringTC(;d=d, medium=medium, bcs=[NeumannBC,])
tc = ScatteringTC(;d=d, medium=medium, bcs=[RobinBC,])
tc = ScatteringTC(;d=d, medium=medium, bcs=[NeumannBC,RobinBC,])
# Transmission type
tp = Idl2TP(;z=1)
tp = SndOrderTP(;z=1,α=h/(2*k^2))
tp = DespresTP(;z=1)
tp = DtN_neighbours_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin)
tp = DtN_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin)
# Solver
solver = Jacobi_S(;tol=1.e-12, maxit=10000, r=0.5, light_mode=false)
solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false)
# DDM type
dd = OnionDDM(;implicit=true)
dd = JunctionsDDM(;implicit=true, precond=true)
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
gid = InputData(m, fullpb, pbs);
ddm = DDM(pbs, gid, dd; to=to);
@timeit to "Resfunc setup" resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact, solver);
@timeit to "Solver" u,x,res = solver(ddm; resfunc=resfunc, to=to);
# Printing some timings
show(to); println("")

# Saving
save_solutions(m, fullpb, solver, u, uexact, prefix, "eraseme");
save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, "eraseme");
