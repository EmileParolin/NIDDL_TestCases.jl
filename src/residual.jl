function get_resfunc(m,fullpb,pbs,ddm,uexact,solver;
                     save_solutions_it=false, prefix="")
    norms = Dict{Symbol,Function}()
    # discrete and continuous L2 norm on the interface traces
    norms[:l2t] = (it) -> norm(ddm.b .- (solver.x .- solver.Ax))
    # continuous HD norm in full domain
    norms[:HD] = (it) -> Inf
    if !(solver.light_mode)
        # "Exact" discrete solutions
        uexactis = [transpose(ld.MΩitoΩ) * uexact for ld in ddm.lds]
        # Discrete solutions using DDM
        ũhis = [() -> ld.Li.ui + ld.Fi for ld in ddm.lds]
        # continuous HD norm in full domain
        AHDis = [A_HDnorm(m,pb.Ω,fullpb) for pb in pbs]
        norms[:HD] = (it) -> sqrt(sum([(AHDi(ũhi() .- uexacti))^2
                                    for (AHDi,ũhi,uexacti) in zip(AHDis,ũhis,uexactis)])
                                /sum([(AHDi(uexacti))^2
                                    for (AHDi,uexacti) in zip(AHDis,uexactis)]))
    end
    resfunc = (it) -> [norms[:l2t](it), norms[:HD](it)]
    # Saving solutions and errors on disk
    if save_solutions_it
        function savefunc(it)
            # Solutions
            uis = [ld.Li.ui + ld.Fi for ld in ddm.lds]
            save_vector_partition(m, [pb.Ω for pb in pbs],
                                    [[("u",toP1(pb, m, pb.Ω, ui))]
                                    for (pb, ui) in zip(pbs, uis)],
                                    prefix*"u_$(it)")
            # Errors
            if !(solver.light_mode)
                uis = [ld.Li.ui + ld.Fi for ld in ddm.lds]
                save_vector_partition(m, [pb.Ω for pb in pbs],
                                        [[("e",toP1(pb, m, pb.Ω, uexacti.-ui))]
                                        for (pb, ui, uexacti) in zip(pbs, uis, uexactis)],
                                        prefix*"e_$(it)")
            end
        end
        return (it) -> (savefunc(it); resfunc(it))
    else
        return resfunc
    end
end
