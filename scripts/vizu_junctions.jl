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
prefix = pwd() * "/data/"

## Helper function
function run(d, k, Nλ, as, tc, nΩ, tp, dd, name)
    # Geometry, mesh and domains
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                interior=false, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix * name);
    Ω = union(Ωs...);
    save_partition(m, [Domain(ω) for ω in Ω], prefix * name * "_partition");
    # Solver
    solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false)
    solver = Jacobi_S(;tol=1.e-12, maxit=50, r=0.5, light_mode=false)
    # Problems
    fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
    # Exact discrete solution
    to = TimerOutput()
    @timeit to "Full problem resolution" uexact = solve(m, fullpb);
    # DDM
    gid = InputData(m, fullpb, pbs);
    ddm = DDM(pbs, gid, dd; to=to);
    @timeit to "Resfunc setup" resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact,
        solver; save_solutions_it=true, prefix=prefix, name=name);
    @timeit to "Solver" u, x, res = solver(ddm; resfunc=resfunc, to=to);
    # Saving and printing some timings
    show(to); println("")
    save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, name);
end

## General parameters
d = 2
pb_type = VectorHelmholtzPb

## Low frequency
k = 1
Nλ = 200
as = [0.5,1,]
medium = AcousticMedium(;k0=k)
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[NeumannBC, RobinBC,])
# Runs Local vs NonLocal
nΩ = 9
postfix = "without_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DespresTP(;z=1)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_Local_$(postfix)")
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
# Runs Onion vs Junctions
nΩ = 10
postfix = "with_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
tp = DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = JunctionsDDM(;implicit=true, precond=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Junctions_NonLocal_$(postfix)")

## Medium frequency
k = 5
Nλ = 80
as = [0.4,1,]
medium = AcousticMedium(;k0=k)
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[NeumannBC, RobinBC,])
# Runs Local vs NonLocal
nΩ = 10
postfix = "without_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DespresTP(;z=1)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_Local_$(postfix)")
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
# Runs Onion vs Junctions
nΩ = 11
postfix = "with_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
tp = DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = JunctionsDDM(;implicit=true, precond=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Junctions_NonLocal_$(postfix)")

## Higher frequency
k = 10
Nλ = 25
as = [0.4,1,]
medium = AcousticMedium(;k0=k)
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[NeumannBC, RobinBC,])
# Runs Local vs NonLocal
nΩ = 8
postfix = "without_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DespresTP(;z=1)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_Local_$(postfix)")
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
# Runs Onion vs Junctions
nΩ = 9
postfix = "with_xpt"
tag = "k$(k)_n$(nΩ)_Nl$(Nλ)_"
tp = DtN_neighbours_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = OnionDDM(;implicit=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Onion_NonLocal_$(postfix)")
tp = DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
dd = JunctionsDDM(;implicit=true, precond=true)
run(d, k, Nλ, as, tc, nΩ, tp, dd, tag*"Junctions_NonLocal_$(postfix)")