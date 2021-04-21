using LaTeXStrings

"""
Structure to hold information gathered from a saved residual file.
"""
struct ResHistory
    resl2::Vector{Float64} # Residual history discrete l2 norm
    resHD::Vector{Float64} # Volume error history
    tp::DataType           # Type of transmission operator
    k::Float64             # Wavenumber
    Nl::Float64            # Nb of points per wavelength
    n::Int64               # Number of sub-domains
    nl::Int64              # Number of cell layers (DtN)
    medium::String         # Label used to characterize medium
    cg_min::Integer        # Minimum number of inner CG (xpts only)
    cg_max::Integer        # Maximum number of inner CG (xpts only)
    cg_sum::Integer        # Total iterations of inner CG (xpts only)
    function ResHistory(r)
        resl2  = r["res"][:,1]
        resHD  = r["res"][:,2]
        tp     = r["tp"]
        k      = r["k"]
        Nl     = r["Nlambda"]
        n      = r["Nomega"]
        nl     = r["nl"]
        medium = r["medium"]
        cg_min = r["cg_min"]
        cg_max = r["cg_max"]
        cg_sum = r["cg_sum"]
        return new(resl2, resHD, tp, k, Nl, n, nl, medium,
                   cg_min, cg_max, cg_sum)
    end
end

function get_res(r::ResHistory, ertype)
    if ertype == :HD
        res = r.resHD
    elseif ertype == :l2
        res = r.resl2
    elseif ertype in [:cg_min, :cg_max, :cg_sum]
        res = r.resl2
    else
        error("Error type $ertype not recognized.")
    end
    return res[Inf .> res .> 0]
end

function tolerance_reached_at(r::ResHistory, tol; ertype=:HD)
    res = get_res(r, ertype)
    if ertype in [:HD, :l2]
        N = sum(res .> tol)
        if N == length(res)
            # Case tolerance not reached within maxiter iterations
            return 0
        else
            return N
        end
    elseif ertype in [:cg_min, :cg_max, :cg_sum]
        if ertype == :cg_min
            return r.cg_min
        elseif ertype == :cg_max
            return r.cg_max
        else
            N = sum(res .> 0)
            return Int64(floor(r.cg_sum / N))
        end
    else
        error("Error type $ertype not recognized.")
    end
end

function get_res_label(r::ResHistory)
    if r.tp == Idl2TP
        return L"${\rm Id}$"
    elseif r.tp == DespresTP
        #return L"${\rm M}$"
        return L"$0^{\mathrm{th}}\mathrm{order}$"
        #return L"Local$~$"
    elseif r.tp == SndOrderTP
        #return L"${\rm K}$"
        return L"$2^{\mathrm{nd}}\mathrm{order}$"
    elseif r.tp == DtN_neighbours_TP
        #return L"${\rm \Lambda}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Aux}}$"
        #return L"Non-local$~$"
    elseif r.tp == DtN_TP
        #return L"${\rm \Lambda}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Aux}}$"
        #return L"Non-local$~$"
    elseif r.tp == HS_TP
        #return L"${\rm W}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Bessel}}$"
    elseif r.tp == invS_TP
        return L"${\rm S}^{-1}$"
    elseif r.tp == EFIE_TP
        #return L"${\rm W}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Bessel}}$"
    elseif r.tp == NL_TP
        #return L"${\rm \Lambda^{*}\Lambda}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Riesz}}$"
    elseif r.tp == LsL_H2D_TP
        #return L"${\rm \Lambda^{*}\Lambda}$"
        return L"$\mathrm{T}_{0}^{\mathrm{Riesz}}$"
    elseif r.tp == Any
        return L"No DDM $~$"
    else
        error("ResHistory Label not recognized.")
    end
end

function generate_conv_plot(files;
                            dir = "./",
                            ertype=:HD, step=1, itmax=2^63-1,
                            marks=["o", "+", "square", "asterisk", "diamond",
                                   "triangle",],
                            colorstyles=["black", "red", "blue", "teal", "cyan",
                                         "orange", "magenta",],
                            styles=["solid" for _ in 1:8],)
    # Printing information
    @info "==> Loading these files"
    for f in files @info "$f" end
    # Loading
    ress = [get_res(ResHistory(load(dir*f*".jld")), ertype) for f in files]
    labs = [get_res_label(ResHistory(load(dir*f*".jld"))) for f in files]
    # Plot
    plts = Array{PGFPlots.Plots.Linear,1}(undef,length(ress))
    for (ii,(res,lab)) in enumerate(zip(ress,labs))
        # No more than itmax
        rs = res[1:min(itmax, length(res))]
        # Removing trailing zeros
        N = length(rs[Inf.>rs.>0])
        # Only step by step
        xs = collect(0:step:N-2)
        ys = res[1:step:N-1]
        # Adding last point at convergence
        push!(xs, N-1)
        push!(ys, rs[N])
        # First residual / error is 1
        ys ./= ys[1]
        # Plot
        plts[ii] = PGFPlots.Plots.Linear(xs, ys,
                                legendentry=lab,
                                mark=marks[1+(ii-1)%length(marks)],
                                style=styles[1+(ii-1)%length(styles)]
                                      *",line width = 0.5pt,"
                                      *colorstyles[1+(ii-1)%length(colorstyles)]
                                )
    end
    # Axis
    a = Axis(plts,
             style="grid=both",
             ymode="log",
             xlabel=L"Iteration number $n$",
             ylabel = L"Relative error $~$",
             #legendPos="outer north east",
             legendPos="north east",
             legendStyle="{font=\\small}" # default is small
             )
    return a
end

function generate_table(data, param_type; tol=1.e-10, ertype=:HD, dir="./")
    if param_type == :epsilon_ || param_type == :mu_
        param_type = :medium
    end
    Nrows, Ncols = size(data)
    # Loading
    ress = Array{ResHistory,length(size(data))}(undef, size(data)...)
    for (id,d) in enumerate(data)
        ress[id] = ResHistory(load(dir*d*".jld"))
    end
    # Operator types, sanity check
    for j in 1:Ncols
        for i in 1:Nrows
            @assert(getproperty(ress[i,j], :tp) == getproperty(ress[1,j], :tp),
                    "Error in input (Operator types).")
        end
    end
    # Parameter under study, sanity check
    for i in 1:Nrows
        for j in 1:Ncols
            p = getproperty(ress[i,j], param_type)
            @assert(p == getproperty(ress[i,end], param_type) || (param_type == :n && p == 0),
                    "Error in input (Parameter).")
        end
    end
    # Row labels
    rlabs = Array{Float64,1}(undef, Nrows)
    for i in 1:Nrows
        prop = getproperty(ress[i,end], param_type)
        # If string, extract numeric (supposed to be at end of string)
        prop_numb = typeof(prop) == String ? tryparse.(Float64, split(prop, "_"))[end] : prop
        rlabs[i] = prop_numb
    end
    # Column labels
    clabs = Array{String,1}(undef, Ncols)
    for j in 1:Ncols
        clabs[j] = get_res_label(ress[1,j])
    end
    # Iteration count
    its = Array{Int64,2}(undef, Nrows, Ncols)
    for i in 1:Nrows
        for j in 1:Ncols
            its[i,j] = tolerance_reached_at(ress[i,j], tol; ertype=ertype)
        end
    end
    return ress, rlabs, clabs, its
end


function generate_param_plot(files;
                             dir="./",
                             param_type=:Nl,
                             tol=1.e-8, ertype=:HD,
                             fullgmres=true,
                             marks=["o", "+", "square", "asterisk", "diamond",
                                    "triangle",],
                             colorstyles=["black", "red", "blue", "teal", "cyan",
                                          "orange", "magenta",],
                             styles=["solid" for _ in 1:8],
                             tll::Vector{TriLogLog}=TriLogLog[],
                             func_on_abscissa=x->x)
    # Printing information
    @info "==> Loading these files"
    for f in files @info "$f" end
    # Loading
    Nrows, Ncols = size(files)
    ress, rlabs, clabs, its = generate_table(files, param_type;
                                             tol=tol, ertype=ertype, dir=dir)
    @show its
    # Plot
    plts = Array{PGFPlots.Plots.Plot,1}(undef,0)
    for j in 1:Ncols
        plt_j = PGFPlots.Plots.Linear(func_on_abscissa(rlabs), its[:,j],
                                      legendentry=clabs[j],
                                      mark=marks[1+(j-1)%length(marks)],
                                      style=styles[1+(j-1)%length(styles)]
                                            *",line width = 0.5pt,"
                                            *colorstyles[1+(j-1)%length(colorstyles)])
        # Sanity check if convergence is not obtained otherwise the legend is
        # messed up
        if sum(its[:,j]) > 0
            push!(plts, plt_j)
        end
    end
    # Slope triangles
    for st in tll append!(plts, add_slope(st)) end
    # Axis
    a = Axis(plts,
             style="grid=both",
             xmode="log",
             ymode="log",
             xlabel="",
             ylabel=L"Iteration count$~$",
             #legendPos="outer north east",
             legendPos="north west",
             legendStyle="{font=\\small}" # default is small
             )
    return a
end