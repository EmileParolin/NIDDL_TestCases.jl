"""
Triangle with nodes:
Bottom left node at (10^α, 10^β)
Top left node at (10^α, 10^(β+n*s))
or
Bottom right node at (10^(α+n), 10^β)
Top right node at (10^(α+n), 10^(β+n*s))
"""
struct TriLogLog
    α
    β
    s # Slope
    n # Streching factor
    flip # Flip the triangle or not
end
TriLogLog(α, β, s, n) = TriLogLog(α, β, s, n, false)


function add_slope(t::TriLogLog)
    if t.flip
        tri = PGFPlots.Plots.Linear([10.0^t.α, 10.0^t.α, 10.0^(t.α+t.n), 10.0^t.α],
                                    [10.0^t.β, 10.0^(t.β+t.n*t.s), 10.0^(t.β+t.n*t.s), 10.0^t.β],
                                    mark="none", style="dashed, area style, black")
        nod = PGFPlots.Plots.Node("\$$(t.s)\$",10.0^(t.α+t.n/2),10.0^(t.β+3*t.n*t.s/4))
    else
        tri = PGFPlots.Plots.Linear([10.0^t.α, 10.0^(t.α+t.n), 10.0^(t.α+t.n), 10.0^t.α],
                                    [10.0^t.β, 10.0^t.β, 10.0^(t.β+t.n*t.s), 10.0^t.β],
                                    mark="none", style="dashed, area style, black")
        nod = PGFPlots.Plots.Node("\$$(t.s)\$",10.0^(t.α+t.n/2),10.0^(t.β+t.n*t.s/4))
    end
    return [tri, nod]
end
