"""Must implement Ω::Domain and BCs::Array{BoundaryCondition,1}"""
abstract type Problem <: AbstractProblem end


function quadrature(Ω::Domain)
    if dim(Ω) == 3
        return tetquad[4]
    elseif dim(Ω) == 2
        return triquad[3]
    elseif dim(Ω) == 1
        return edgquad[2]
    else
        error("No quadrature is defined for a domain with dimension $(dim(Ω))")
    end
end
toP1(pb::Problem,m::Mesh,Ω::Domain,u) = u


function get_mass_matrix(m::Mesh,Ω::Domain,pb::Problem; coef=missing)
    # Quadrature
    q = quadrature(Ω)
    W = ismissing(coef) ? weight_matrix(m,Ω,q) : weight_matrix(m,Ω,q; coef=coef)
    # Finite Elements types
    u = unknown_fe_type(Ω,pb)
    # Finite Elements definitions
    Mu = assemble(u,m,Ω,q)
    # Finite Elements matrices
    u_u = femdot(Mu,W,Mu)
    return u_u
end


function get_matrix_building_blocks(m::Mesh,Ω::Domain,pb::Problem;
                                    coefM=missing, coefK=missing)
    # Quadrature
    q = quadrature(Ω)
    if ismissing(coefM) && ismissing(coefK)
        WM = weight_matrix(m,Ω,q)
        WK = WM
    else
        WM = weight_matrix(m,Ω,q; coef=coefM)
        WK = weight_matrix(m,Ω,q; coef=coefK)
    end
    # Finite Elements types
    u = unknown_fe_type(Ω,pb)
    Du = D_unknown_fe_type(Ω,pb)
    # Finite Elements definitions
    Mu = assemble(u,m,Ω,q)
    MDu = assemble(Du,m,Ω,q)
    # Finite Elements matrices
    u_u = femdot(Mu,WM,Mu)
    Du_Du = femdot(MDu,WK,MDu)
    return u_u, Du_Du
end


function apply_bc(m::Mesh,pb::Problem,A)
    # Applying physical boundary conditions
    for bc in filter(bc->typeof(bc) <: PhysicalBC, pb.BCs)
        [@assert tag in tags(boundary(pb.Ω)) for tag in tags(bc.Γ)]
        A = apply(A,m,pb,bc)
    end
    # Applying transmission boundary conditions
    for bc in filter(bc->typeof(bc) <: TransmissionBC, pb.BCs)
        [@assert tag in tags(boundary(pb.Ω)) for tag in tags(bc.Γ)]
        A = apply(A,m,pb,bc)
    end
    return A
end


function get_matrix(m::Mesh,pb::Problem)
    u_u, Du_Du = get_matrix_building_blocks(m,pb.Ω,pb;
                                            coefM=bcoef(pb.medium),
                                            coefK=acoef(pb.medium))
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    A = RΩ * (Du_Du - u_u) * transpose(RΩ)
    A = apply_bc(m, pb, A)
    return A
end


function get_matrix_no_transmission_BC(m::Mesh,pb::Problem)
    pb_noTBC = typeof(pb)(pb.medium, pb.Ω, filter(bc->typeof(bc)<:PhysicalBC, pb.BCs))
    return get_matrix(m,pb_noTBC)
end


function get_rhs(m::Mesh,pb::Problem)
    # Initialisation
    b = zeros(Complex{Float64}, number_of_elements(m, pb.Ω, dofdim(pb)))
    # Applying physical boundary conditions
    for bc in filter(bc->typeof(bc) <: PhysicalBC, pb.BCs)
        [@assert tag in tags(boundary(pb.Ω)) for tag in tags(bc.Γ)]
        b = rhs(b,m,pb,bc)
    end
    # Applying transmission boundary conditions
    for bc in filter(bc->typeof(bc) <: TransmissionBC, pb.BCs)
        [@assert tag in tags(boundary(pb.Ω)) for tag in tags(bc.Γ)]
        b = rhs(b,m,pb,bc)
    end
    return b
end


function solve(m::Mesh,pb::Problem)
    K = get_matrix(m,pb)
    f = get_rhs(m,pb)
    KLU = factorize(K)
    u = KLU \ f
    return u
end


function solve_gmres(m::Mesh,pb::Problem; tol=1.e-3, maxit=100, restart=20,
                     light_mode=true, uexact=0)
    K = get_matrix(m,pb)
    f = get_rhs(m,pb)
    u = zeros(Complex{Float64}, length(f))
    # Volume energy norm
    nrg_norm = u -> Inf
    if !light_mode
        AHD = A_HDnorm(m,pb.Ω,pb)
        nrg_norm = u -> (AHD(u .- uexact) / AHD(uexact))
    end
    # GMRES iterator
    g = IterativeSolvers.gmres_iterable!(u, K, f; tol=tol, maxiter=maxit,
                                        restart=restart, light_mode=light_mode)
    # Residual (or other types of error)
    res = zeros(Float64,maxit,2).+Inf # for convergence plots
    res[it,1] = resl2 # exact l2 residual
    res[it,2] = nrg_norm(g.xbis)
    @info "Iteration $(it) at $(res[it,:])"
    return u, res
end


function cond(m::Mesh,pb::Problem)
    K = get_matrix(m,pb)
    c = cond(Array(K))
    return c
end

###############################
# Matrices of scalar products #
###############################
abstract type A_norm end

struct A_L2norm <: A_norm
    A
    function A_L2norm(m::Mesh,Ω::Domain,pb::Problem)
        u_u, Du_Du = get_matrix_building_blocks(m,pb.Ω,pb;
                                                coefM=bcoef(pb.medium),
                                                coefK=acoef(pb.medium))
        RΩ = restriction(m,Ω,dofdim(pb))
        return new(RΩ * u_u * transpose(RΩ))
    end
end
(n::A_L2norm)(x) = sqrt(real(dot(x, n.A * x)))

struct A_HDseminorm <: A_norm
    A
    function A_HDseminorm(m::Mesh,Ω::Domain,pb::Problem)
        u_u, Du_Du = get_matrix_building_blocks(m,pb.Ω,pb;
                                                coefM=bcoef(pb.medium),
                                                coefK=acoef(pb.medium))
        RΩ = restriction(m,Ω,dofdim(pb))
        return new(RΩ * Du_Du * transpose(RΩ))
    end
end
(n::A_HDseminorm)(x) = sqrt(real(dot(x, n.A * x)))

struct A_HDnorm <: A_norm
    A
    function A_HDnorm(m::Mesh,Ω::Domain,pb::Problem)
        u_u, Du_Du = get_matrix_building_blocks(m,pb.Ω,pb;
                                                coefM=bcoef(pb.medium),
                                                coefK=acoef(pb.medium))
        RΩ = restriction(m,Ω,dofdim(pb))
        return new(RΩ * (u_u + Du_Du) * transpose(RΩ))
    end
end
(n::A_HDnorm)(x) = sqrt(real(dot(x, n.A * x)))
