using Test
using LinearAlgebra
using Random
using NIDDL_FEM
using NIDDL
using NIDDL_TestCases
using GmshSDK

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
                        gid = StandardInputData(m, fullpb, pbs)
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


# Warning iteration count are dependent on the GMSH version.
Nits = Dict((2, AcousticMedium, :basic)
            =>[14013, 21, 14013, 21, 290, 21, 290, 21, 67, 17, 67, 17, 17709,
               73, 17709, 73, 301, 55, 301, 55, 71, 21, 71, 21, 7976, 17, 7976,
               17, 338, 17, 338, 17, 71, 17, 71, 17],
            (2, AcousticMedium, :xpts)
            =>[14875, 50, 14875, 50, 294, 49, 294, 49, 116, 21, 116, 21, 19039,
               105, 19039, 105, 257, 65, 257, 65, 106, 26, 106, 26, 12697, 61,
               12697, 61, 293, 43, 293, 43, 79, 22, 79, 22],
            (3, AcousticMedium, :basic)
            =>[70271, 167, 70271, 167, 878, 54, 878, 54, 194, 43, 194, 43,
               10749, 298, 10749, 298, 907, 89, 907, 89, 217, 58, 217, 58,
               34967, 183, 34967, 183, 515, 76, 515, 76, 242, 67, 242, 67],
            (3, AcousticMedium, :xpts)
            =>[75726, 201, 75726, 201, 925, 162, 925, 162, 359, 132, 359, 132,
               11432, 453, 11432, 453, 970, 179, 970, 179, 389, 149, 389, 149,
               37193, 269, 37193, 269, 1141, 180, 1141, 180, 314, 133, 314,
               133],
            (3, ElectromagneticMedium, :basic)
            =>[276875, 145, 276875, 145, 791, 73, 791, 73, 75, 21, 75, 21,
               92769, 131, 92769, 131, 873, 99, 873, 99, 85, 26, 85, 26, 74200,
               132, 74200, 132, 889, 101, 889, 101, 77, 30, 77, 30],
            (3, ElectromagneticMedium, :xpts)
            =>[279445, 181, 279445, 181, 797, 86, 797, 86, 71, 26, 71, 26, 95182,
             166, 95182, 166, 734, 100, 734, 100, 81, 30, 81, 30, 75729, 169,
             75729, 169, 855, 110, 855, 110, 77, 32, 77, 32],
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
                pb_type = typeof(medium) == AcousticMedium ? HelmholtzPb : MaxwellPb
                tcs = [ScatteringTC(;d=d, pb_type, medium=medium, θ0=π/2, ϕ0=0, bcs=[RobinBC,]),]
                solvers = [Jacobi_S(;tol=1e-10, maxit=1000000, r=0.5, light_mode=false),
                           GMRES_S(;tol=1e-10, maxit=100000, light_mode=false), ]
                local Πinvolution, Nits_test, solution_found
                @testset "Exchange type : standard" begin
                    tps = [DespresTP(;z=1), SndOrderTP(;z=1,α=1/(2*k^2)),
                           DtN_neighbours_TP(;z=1, pb_type=pb_type,
                                             medium=dissipative_medium(medium),
                                             fbc=:robin),]
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
                           DtN_TP(;z=1, pb_type=pb_type,
                                  medium=dissipative_medium(medium),
                                  fbc=:robin),]
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
