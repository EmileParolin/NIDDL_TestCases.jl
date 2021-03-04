using Revise, Pkg, Test
Pkg.activate("./")
using LinearAlgebra, SparseArrays, SuiteSparse, SharedArrays, Distributed
using LinearMaps, IterativeSolvers, TimerOutputs
using NIDDL_FEM, NIDDL, NIDDL_TestCases
using WriteVTK
prefix = pwd() * "/data/"

##
d = 2
k = 1
Nλ = 50
as = [1,]
nΩ = 5
name = "eraseme"
g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                interior=true, nΩ=nΩ, mode=:metis, nl=0, layer_from_PBC=true)
h = 2π / abs(k) / max(5, Nλ);
m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix * name);
Ω = union(Ωs...);
save_partition(m, [Domain(ω) for ω in Ω], prefix * name * "_partition");
# Test case
medium = AcousticMedium(;k0=k)
pb_type = HelmholtzPb
tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])

## Skeleton DDM
Σsolver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false);
Σdd = JunctionsDDM(;implicit=true, inductive=false, precond=true);
Σtp = Idl2TP(;z=1);
Σfullpb, Σpbs = get_skeleton_problems(m, Ω, Ωs, pb_type, k, Σtp);
Σgid = StandardInputData(m, Σfullpb, Σpbs);
Σddm = DDM(Σpbs, Σgid, Σdd);
Σuexact = solve(m, Σfullpb);
Σresfunc = get_resfunc(m, Σfullpb, Σpbs, Σddm, Σuexact, Σsolver);

## DDM
solver = Jacobi_S(;tol=1.e-12, maxit=10000, r=0.5, light_mode=false)
solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false)
dd = JunctionsDDM(;implicit=true, inductive=true, precond=true)
tp = SndOrderTP(;z=1)
fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
gid = StandardInputData(m, fullpb, pbs);
gid = InductiveInputData(m, fullpb, pbs, Σfullpb, Σpbs, Σgid, Σsolver, Σddm, Σresfunc);
ddm = DDM(pbs, gid, dd);
uexact = solve(m, fullpb);
resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact, solver);
u,x,res = solver(ddm; resfunc=resfunc);

##
ddm.gd.Σfullpb.b0 = zeros(Complex{Float64}, length(Σfullpb.b0))
for (σpb, MΣsttoσi) in zip(ddm.gd.Σpbs, ddm.gd.MΣsttoσis)
    σpb.b0 = randn(length(σpb.b0))
    ddm.gd.Σfullpb.b0 += transpose(MΣsttoσi) * σpb.b0
end
Σuexact = solve(m, ddm.gd.Σfullpb);
ddm.gd.Σresfunc = get_resfunc(m, ddm.gd.Σfullpb, ddm.gd.Σpbs, ddm.gd.Σddm, Σuexact, ddm.gd.Σsolver);
Σu,Σx,Σres = ddm.gd.Σsolver(ddm.gd.Σddm; resfunc=ddm.gd.Σresfunc);