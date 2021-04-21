using Revise
using Pkg
Pkg.activate("./")
#Pkg.update("NIDDL_FEM")
#Pkg.update("NIDDL")
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
#using Test
using JLD
#using PGFPlots
prefix = pwd() * "/data/"
#include("./scripts/TriLogLog.jl")
#include("./scripts/postprod.jl")

##
function daidai(; d = 2, k = 1, Nλ = 20, nΩ = 4, aR = 1, name="eraseme", op=:Id)
    pb_type = d == 2 ? VectorHelmholtzPb : MaxwellPb
    as = [aR,]
    medium = d == 2 ? AcousticMedium(;k0=k) : ElectromagneticMedium(;k0=k)
    tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
    tp = op == :Id ? DespresTP(;z=1) : DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
    dd = JunctionsDDM(;implicit=true, precond=true)
    # Geometry, mesh and domains
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                interior=true, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h;);
    Ω = union(Ωs...);
    # Solver
    solver = Jacobi_S(;tol=1.e-12, maxit=50, r=0.5, light_mode=false)
    solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false)
    # Problems
    fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
    # Exact discrete solution
    to = TimerOutput()
    @timeit to "Full problem resolution" uexact = solve(m, fullpb);
    # DDM
    gid = InputData(m, fullpb, pbs);
    ddm = DDM(pbs, gid, dd; to=to);
    @timeit to "Resfunc setup" resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact,
        solver; save_solutions_it=false, prefix=prefix);
    @timeit to "Solver" u, x, res = solver(ddm; resfunc=resfunc, to=to);
    # Output
    #save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, name);
    JLD.save(prefix*name*".jld",
        "res", res, "tp", typeof(tp), "k", medium.k0, "Nlambda", Nλ, "Nomega", nΩ,
        "medium", medium.name, "nl", g.nl, "cg_min", ddm.gd.cg_min, "cg_max",
        ddm.gd.cg_max, "cg_sum", ddm.gd.cg_sum,)
    return u, x, res, ddm
end

##
d = 2; k = 2; Nλ = 20
Js = 2 .^ collect(10:-1:0)
for J in Js
    aR = Int64(floor(1000*( 1. * J^(1/d) )))/1000
    area = π * aR^d
    areaj = area / J
    h = 2π/k / Nλ
    ndofj = areaj / h^d
    @info "J = $(J) \t| R = $(aR) \t| h = $(h) \t| Ndofj = $(ndofj)"
    name = replace("weak_scaling_$(d)D_k$(k)_Nl$(Nλ)_J$(J)_a$(aR)", "."=>"d");
    u, x, res, ddm = daidai(;d=d, k=k, Nλ=Nλ, nΩ=J, aR=aR, name=name*"_Despres", op=:Id);
    u, x, res, ddm = daidai(;d=d, k=k, Nλ=Nλ, nΩ=J, aR=aR, name=name*"_DtN",     op=:DtN);
end
