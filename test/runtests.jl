using Test
using LinearAlgebra
using Random
using NIDDL_FEM
using NIDDL
using NIDDL_TestCases
include("../src/gmsh.jl")

function andiamo(;geos=Geometry[], Nλs=[20,], tcs=TestCase[],
                 tps=TransmissionParameters[], solvers=Solver[],
                 dds=DDM_Type[])
    Nits_test = Int64[]
    Πinvolution = Bool[]
    solution_found = Bool[]
    for g in geos
        for Nλ in Nλs
            # Mesh and domains
            h = 2π/abs(maximum([tc.medium.k0 for tc in tcs])) / max(10, Nλ)
            m, Ωs, Γs = get_mesh_and_domains(g, h)
            for tc in tcs
                for tp in tps
                    for dd in dds
                        # Problems
                        fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd)
                        # Exact discrete solution
                        uexact = solve(m,fullpb)
                        # DDM
                        gid = InputData(m, fullpb, pbs)
                        ddm = DDM(pbs, gid, dd);
                        for solver in solvers
                            # Testing that we find the correct solution
                            solver.Ax .= 0
                            solver.x .= 0
                            resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact, solver);
                            u,x,res = solver(ddm; resfunc=resfunc);
                            push!(solution_found, norm(uexact .- u) / norm(uexact) < 1.e-8)
                            # Iteration count
                            push!(Nits_test, length(res[Inf.>res[:,1].>0,1]))
                            # Exchange operator is an involution
                            y = randn(length(x)) .+ im .* randn(length(x))
                            push!(Πinvolution, norm(ddm.A.Π(copy(ddm.A.Π(y))) .- y) < 1.e-8)
                        end
                    end
                end
            end
        end
    end
    return Πinvolution, Nits_test, solution_found
end


Nits = Dict((2, AcousticMedium, :basic)
            =>[13334, 21, 13334, 21, 306, 21, 306, 21, 67, 17, 67, 17, 6708,
               21, 6708, 21, 660, 21, 660, 21, 71, 17, 71, 17, 9127, 17, 9127,
               17, 227, 17, 227, 17, 72, 17, 72, 17],
            (2, AcousticMedium, :xpts)
            =>[14201, 49, 14201, 49, 311, 51, 311, 51, 116, 21, 116, 21, 7105,
               40, 7105, 40, 719, 61, 719, 61, 112, 21, 112, 21, 11140, 64,
               11140, 64, 201, 40, 201, 40, 79, 22, 79, 22],
            (3, AcousticMedium, :basic)
            =>[46518, 136, 46518, 136, 996, 55, 996, 55, 195, 41, 195, 41,
               6040, 262, 6040, 262, 694, 73, 694, 73, 231, 51, 231, 51, 9990,
               139, 9990, 139, 429, 69, 429, 69, 214, 61, 214, 61],
            (3, AcousticMedium, :xpts)
            =>[49990, 181, 49990, 181, 1065, 165, 1065, 165, 368, 124, 368,
               124, 6419, 319, 6419, 319, 749, 158, 749, 158, 384, 141, 384,
               141, 8375, 172, 8375, 172, 926, 189, 926, 189, 291, 124, 291,
               124],
            (3, ElectromagneticMedium, :basic)
            =>[209747, 131, 209747, 131, 471, 54, 471, 54, 75, 19, 75, 19,
               43678, 101, 43678, 101, 1005, 107, 1005, 107, 87, 25, 87, 25,
               28240, 117, 28240, 117, 842, 101, 842, 101, 77, 31, 77, 31],
            (3, ElectromagneticMedium, :xpts)
            =>[212398, 161, 212398, 161, 472, 62, 472, 62, 70, 26, 70, 26,
               44339, 128, 44339, 128, 843, 105, 843, 105, 81, 29, 81, 29,
               38336, 141, 38336, 141, 827, 115, 827, 115, 77, 32, 77, 32],
           )


@testset "All tests" begin
    @testset "Non-regression" begin
        k = 1
        for (d,medium) in [(2,AcousticMedium(;k0=k)),
                           (3,AcousticMedium(;k0=k)),
                           (3,ElectromagneticMedium(;k0=k)),
                          ]
            @testset "$(typeof(medium)) $(d)D" begin
                Nλs = [20,]
                geos = [LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[0.5, 1,],
                                interior=true, nΩ=2, mode=:cad, nl=0),
                        LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[0.5, 1,],
                                interior=true, nΩ=2, mode=:geo, nl=0),
                        LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[     1,],
                                interior=true, nΩ=2, mode=:metis, nl=0), ]
                tcs = [ScatteringTC(;d=d, medium=medium, θ0=π/2, ϕ0=0, bcs=[RobinBC,]),]
                solvers = [Jacobi_S(;tol=1e-10, maxit=1000000, r=0.5, light_mode=false),
                           GMRES_S(;tol=1e-10, maxit=100000, light_mode=false), ]
                local Πinvolution, Nits_test, solution_found
                @testset "Exchange type : standard" begin
                    tps = [DespresTP(;z=1), SndOrderTP(;z=1,α=1/(2*k^2)),
                           DtN_neighbours_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin),]
                    dds = [OnionDDM(;implicit=false),
                           OnionDDM(;implicit=true),]
                    info = andiamo(geos=geos, Nλs=Nλs, tcs=tcs, tps=tps,
                                   solvers=solvers, dds=dds);
                    Πinvolution, Nits_test, solution_found = info
                    @show Nits_test
                    bool = Nits[(d, typeof(medium), :basic)] .== Nits_test
                    for (is, solver) in enumerate(["Richardson", "GMRES"])
                        @testset "$solver" begin
                            @testset "Agreement Implicit/Explicit" begin
                                n1 = Nits_test[is:2:end][1:2:end]
                                n2 = Nits_test[is:2:end][2:2:end]
                                for (it, tc) in enumerate(["0th", "2nd", "NL"])
                                    @testset "$tc" begin
                                        for (ig, geo) in enumerate(["cad", "geo", "metis"])
                                            @testset "$geo" begin
                                                @test n1[it:3:end][ig] == n2[it:3:end][ig]
                                            end
                                        end
                                    end
                                end
                            end
                            for (im, mode) in enumerate(["Implicit", "Explicit"])
                                @testset "$mode" begin
                                    for (it, tc) in enumerate(["0th", "2nd", "NL"])
                                        @testset "$tc" begin
                                            for (ig, geo) in enumerate(["cad", "geo", "metis"])
                                                @testset "$geo" begin
                                                    @testset "Exchange involution" begin
                                                        @test Πinvolution[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                    @testset "Solution found" begin
                                                        @test solution_found[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                    @testset "Iteration count" begin
                                                        @test bool[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                @testset "Exchange type : projection" begin
                    tps = [DespresTP(;z=1), SndOrderTP(;z=1,α=1/(2*k^2)),
                           DtN_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin),]
                    dds = [JunctionsDDM(;implicit=false, precond=true),
                           JunctionsDDM(;implicit=true,  precond=true),]
                    info = andiamo(geos=geos, Nλs=Nλs, tcs=tcs, tps=tps,
                                   solvers=solvers, dds=dds);
                    Πinvolution, Nits_test, solution_found = info
                    @show Nits_test
                    bool = Nits[(d, typeof(medium), :xpts)] .== Nits_test
                    for (is, solver) in enumerate(["Richardson", "GMRES"])
                        @testset "$solver" begin
                            @testset "Agreement Implicit/Explicit" begin
                                n1 = Nits_test[is:2:end][1:2:end]
                                n2 = Nits_test[is:2:end][2:2:end]
                                for (it, tc) in enumerate(["0th", "2nd", "NL"])
                                    @testset "$tc" begin
                                        for (ig, geo) in enumerate(["cad", "geo", "metis"])
                                            @testset "$geo" begin
                                                @test n1[it:3:end][ig] == n2[it:3:end][ig]
                                            end
                                        end
                                    end
                                end
                            end
                            for (im, mode) in enumerate(["Implicit", "Explicit"])
                                @testset "$mode" begin
                                    for (it, tc) in enumerate(["0th", "2nd", "NL"])
                                        @testset "$tc" begin
                                            for (ig, geo) in enumerate(["cad", "geo", "metis"])
                                                @testset "$geo" begin
                                                    @testset "Exchange involution" begin
                                                        @test Πinvolution[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                    @testset "Solution found" begin
                                                        @test solution_found[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                    @testset "Iteration count" begin
                                                        @test bool[is:2:end][im:2:end][it:3:end][ig]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
