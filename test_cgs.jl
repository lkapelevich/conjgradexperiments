import Random
import ForwardDiff
import DiffResults
using LinearAlgebra
using DataFrames
using CSV
using Statistics
using Printf
using DoubleFloats
include("conjutilities.jl")
# include("grad_hess.jl")
Random.seed!(1)
num_samples = 10
cone_ds = [20, 40, 60]
offsets = 10.0 .^ collect(-5:-1)

##
abstract type Cone end
struct SumLog <: Cone
    d::Int
end
struct HypoPower <: Cone
    α::Vector{Float64}
end
HypoPower(d::Int) = (α = rand(d); α /= sum(α); HypoPower(α))
struct HypoGeom <: Cone
    d::Int
end
struct RadialPower <: Cone
    α::Vector{Float64}
end
RadialPower(d::Int) = (α = rand(d); α /= sum(α); RadialPower(α))
struct RadialGeom <: Cone
    d::Int
end
struct InfinityNorm <: Cone
    d::Int
end

##
function barrier(::SumLog)
    function bar(uvw)
        (u, v, w) = (uvw[1], uvw[2], uvw[3:end])
        return -log(v * sum(log(wi / v) for wi in w) - u) - sum(log, w) - log(v)
    end
    return bar
end
function barrier(cone::HypoPower)
    α = cone.α
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return -log(exp(sum(αi * log(wi) for (αi, wi) in zip(α, w))) - u) -
            sum(log, w)
    end
    return bar
end
function barrier(cone::HypoGeom)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return -log(exp(sum(log(wi) / cone.d for wi in w)) - u) - sum(log, w)
    end
    return bar
end
function barrier(cone::RadialPower)
    α = cone.α
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return -log(exp(sum(2αi * log(wi) for (αi, wi) in zip(α, w))) - u^2) -
            sum((1 - αi) * log(wi) for (αi, wi) in zip(α, w))
    end
    return bar
end
function barrier(cone::RadialGeom)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        d = cone.d
        return -log(exp(sum(2 / d * log(wi) for wi in w)) - u^2) - (d - 1) / d *
            sum(log(wi) for wi in w)
    end
    return bar
end
function barrier(cone::InfinityNorm)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return sum(-log(u^2 - wi^2) for wi in w) + (cone.d - 1) * log(u)
    end
    return bar
end

init_guess(cone::SumLog) = vcat(-1, 1, ones(cone.d))
init_guess(cone::HypoPower) = vcat(-1, ones(length(cone.α)))
init_guess(cone::HypoGeom) = vcat(-1, ones(cone.d))
init_guess(cone::RadialPower) = vcat(0, sqrt.(1 .+ cone.α))
init_guess(cone::RadialGeom) = vcat(0, fill(sqrt.(1 + inv(cone.d)), cone.d))
init_guess(cone::InfinityNorm) = vcat(1, zeros(cone.d))

function dual_point(cone::SumLog, offset::Float64)
    d = cone.d
    r = rand(d)
    p = -rand()
    phi = (p * sum(log(-ri / p) for ri in r) + p * d)
    q = phi * (1 + sign(phi) * offset)
    return vcat(p, q, r)
end
function dual_point(cone::HypoPower, offset::Float64)
    α = cone.α
    r = rand(length(α))
    phi = -exp(sum(αi * log(ri / αi) for (ri, αi) in zip(r, α)))
    p = phi * (1 - offset)
    return vcat(p, r)
end
dual_point(cone::HypoGeom, offset::Float64) = dual_point(HypoPower(cone.d), offset)
function dual_point(cone::RadialPower, offset::Float64)
    α = cone.α
    r = rand(length(α))
    # only look at the p > 0 case, but that's OK because residuals aren't
    # going to behave differently in the other direction
    phi = exp(sum(αi * log(ri / αi) for (ri, αi) in zip(r, α)))
    p = phi * (1 - offset)
    return vcat(p, r)
end
dual_point(cone::RadialGeom, offset::Float64) = dual_point(RadialPower(cone.d), offset)
function dual_point(cone::InfinityNorm, offset::Float64)
    r = randn(cone.d)
    p = sum(abs, r) * (1 + offset)
    return vcat(p, r)
end

function grad_hess(cone::Cone, s)
    bar = barrier(cone)
    # s = BigFloat.(s)
    result = DiffResults.HessianResult(s)
    return ForwardDiff.hessian!(result, bar, s);
end

function dual_grad(cone::SumLog, pqr)
    d = cone.d
    (p, q, r) = (pqr[1], pqr[2], pqr[3:end])
    dual_ϕ = sum(log(r_i / -p) for r_i in r)
    β = 1 + d - q / p + dual_ϕ
    bomega = d * wrightomega(β / d - log(d))
    @assert bomega + d * log(bomega) ≈ β

    cgp = (-d - 2 + q / p + 2 * bomega) / (p * (1 - bomega))
    cgq = -1 / (p * (1 - bomega))
    cgr = bomega ./ r / (1 - bomega)
    return (vcat(cgp, cgq, cgr), 0)
end
function dual_grad(cone::HypoPower, pr)
    (p, r) = (pr[1], pr[2:end])
    α = cone.α
    d = length(α)
    sumlog = sum(α_i * log(r_i) for (α_i, r_i) in zip(α, r))

    h(y) = sum(αi * log(y - p * αi) for αi in α) - sumlog
    hp(y) = sum(αi / (y - p * αi) for αi in α)
    lower = 0.0
    upper = exp(sumlog) + p / d
    # (new_bound, iter) = rootnewton(h, hp, lower = lower, upper = upper)
    (new_bound, iter) = rootnewton(h, hp, init = 0.0, increasing = true)

    dual_g_ϕ = inv(new_bound)
    cgp = -inv(p) - dual_g_ϕ
    cgr = (p * dual_g_ϕ * α .- 1) ./ r
    return (vcat(cgp, cgr), iter)
end
function dual_grad(cone::HypoGeom, pr)
    d = cone.d
    (p, r) = (pr[1], pr[2:end])
    dual_ϕ = exp(sum(log, r) / d)
    dual_ζ = dual_ϕ + p / d
    cgp = -inv(p) - inv(dual_ζ)
    cgr = -dual_ϕ / dual_ζ ./ r
    return (vcat(cgp, cgr), 0)
end
function dual_grad(cone::RadialPower, pr)
    (p, r) = (pr[1], pr[2:end])
    α = cone.α
    d = length(α)
    if iszero(p)
        cgp = 0
        cgr = -(α .+ 1) ./ r
    else
        log_phi_r = 2 * sum(αi * log(ri) for (αi, ri) in zip(α, r))
        phi_r = exp(log_phi_r)
        phi_αr = exp(sum(αi * log(ri / αi) for (αi, ri) in zip(α, r)))
        inner_bound = -1 / p - (1 + sign(p) * 1 / p * sqrt(phi_r * (d^2 / p^2 *
            phi_r + d^2 - 1))) / (p / d - d * phi_r / p)
        gamma = abs(p) / phi_αr
        outer_bound = (1 + d) * gamma / (1 - gamma) / p
        h(y) = 2 * sum(αi * log(2 * αi * y^2 + (1 + αi) * 2 * y / p) for αi in α) -
            log_phi_r - log(2 * y / p + y^2) - 2 * log(2 * y / p)
        hp(y) = 2 * sum(αi^2 / (αi * y + (1 + αi) / p) for αi in α) -
            2 * (y + 1 / p) / y / (y + 2 / p)
        # For this cone we start at a different point to "inner_bound" (which
        # guarantees quadratic convergence) because "outer_bound" gives fewer
        # iterations in practice. With `rootnewton` we never step outside of
        # (inner_bound, outer_bound).
        (cgp, iter) = rootnewton(h, hp, lower = inner_bound, upper = outer_bound,
            init = outer_bound, increasing = false)
        cgr = -(α * (1 + p * cgp) .+ 1) ./ r
    end
    return (vcat(cgp, cgr), iter)
end
function dual_grad(cone::RadialGeom, pr)
    d = cone.d
    (p, r) = (pr[1], pr[2:end])
    if iszero(p)
        ctp = 0
        cgr = -(inv(d) + 1) ./ r
    else
        phi_r = exp(2 * sum(log(ri) / d for ri in r))
        cgp = -1 / p - (1 + sign(p) * 1 / p * sqrt(phi_r * ((d / p)^2 * phi_r +
            d^2 - 1))) / (p / d - d * phi_r / p)
        cgr = -((1 + p * cgp) / d + 1) ./ r
    end
    return (vcat(cgp, cgr), 0)
end
function dual_grad(cone::InfinityNorm, pr)
    d = cone.d
    (p, r) = (pr[1], pr[2:end])
    dual_zeta = p - sum(abs, r)
    h(y) = p * y + sum(sqrt(1 + abs2(ri * y)) for ri in r) + 1
    hp(y) = p + y * sum(abs2(ri) / sqrt(1 + abs2(ri * y)) for ri in r)
    # lower = -(d + 1) / dual_zeta
    upper = min(-inv(dual_zeta), -(d + 1) / p)
    (cgp, iter) = rootnewton(h, hp, init = upper)

    cgr = copy(r)
    for i in eachindex(r)
        if abs(r[i]) .< eps()
            cgr[i] = 0
        else
            cgr[i] = (-1 + sqrt(1 + abs2(r[i] * cgp))) / r[i]
        end
    end
    return (vcat(cgp, cgr), iter)
end

function naive_dual_grad(cone, z)
    curr = init_guess(cone)
    derivs = grad_hess(cone, curr)
    quad_bound = 0.35
    (g, H) = (DiffResults.gradient(derivs), DiffResults.hessian(derivs))
    H_fact = cholesky(Symmetric(H))
    r = z + g
    Hir = H_fact \ r
    n = n_prev = sqrt(dot(r, Hir))
    #
    # Hiz = H_fact \ z
    # Hir = Hiz - curr
    # n = n_prev = sqrt(length(z) - 2 * dot(curr, z) + dot(z, Hiz))
    max_iter = 400
    iter = 1
    resid_path = []

    while n > 1000eps()
        α = (n > quad_bound ? inv(1 + n) : 1)
        curr -= Hir * α
        push!(resid_path, dot(z, -curr) + length(z))
        derivs = grad_hess(cone, curr)
        (g, H) = (DiffResults.gradient(derivs), DiffResults.hessian(derivs))
        H_fact = cholesky(Symmetric(H), check = false)
        if !issuccess(H_fact)
            H_fact = bunchkaufman(Symmetric(H))
        end
        r = z + g
        Hir = H_fact \ r
        n2 = dot(r, Hir)
        # Hiz = H_fact \ z
        # Hir = Hiz - curr
        # n2 = length(z) - 2 * dot(curr, z) + dot(z, Hiz)
        n2 < -sqrt(eps()) && break # couple crazy hypogeom cases
        n = sqrt(abs(n2))
        iter += 1
        # slow progress (numerical)
        if (n_prev < quad_bound) && (n > 1000(n_prev / (1 - n_prev))^2)
            break
        end
        n_prev = n
        (iter == max_iter) && break
    end
    return (-curr, iter, resid_path)
end

##
cone_ts = [
    SumLog
    HypoPower
    HypoGeom
    RadialPower
    RadialGeom
    InfinityNorm
    ]

function printx(x::Float64)
    if abs(x) < 10
        return @sprintf("%.1f", x)
    else
        return @sprintf("%.0f.", x)
    end
end

function get_resids()
    reio = open("tex/resid.tex", "w")
    itio = open("tex/iters.tex", "w")
    sep = " & "
    for d in cone_ds
        results = DataFrame(
            cone_d = Int[],
            cone_type = Symbol[],
            offset = Float64[],
            trial = Int[],
            residdirect = Float64[],
            residnaive = Float64[],
            newtoniters = Int[],
            directiters = Int[],
            )

        for C in cone_ts, o in offsets, i in 1:num_samples
            Random.seed!(i)
            cone = C(d)
            z = dual_point(cone, o)
            (g, iter) = dual_grad(cone, z)
            resid = abs(dot(z, g) + length(z)) # if -g slightly out of primal cone then can be numerically negative
            # @show resid
            (naive_g, naive_iter, resid_path) = naive_dual_grad(cone, z)
            naive_resid = abs(dot(z, naive_g) + length(z))
            push!(results, (d, nameof(C), o, i, resid,
                naive_resid, naive_iter, iter))
            if i == 1 && d == 60
                @show nameof(C), resid
                resid_df = DataFrame(resid = resid_path)
                resids_name = lowercase(string(nameof(C))) * string(Int(-log10(o)))
                CSV.write(joinpath("csvs", resids_name * ".csv"), resid_df)
            end
        end

        agg = combine(groupby(results, [:cone_d, :cone_type, :offset]),
            :residdirect => mean => :residdirectmean,
            :residnaive => mean => :residnaivemean,
            :newtoniters => mean => :newtonitersmean,
            :directiters => mean => :directitersmean,
            )

        for subdf in groupby(agg, [:cone_d, :offset])
            for r in eachrow(subdf)
                print(reio, sep * printx(log10(abs(r.residnaivemean))) * sep *
                    printx(log10(abs(r.residdirectmean))))
                print(itio, sep * printx(r.newtonitersmean) *
                    sep * printx(r.directitersmean))
            end
            print(reio, "\\\\ \n")
            print(itio, "\\\\ \n")
        end

        CSV.write("csvs/cgs_$(d).csv", agg)
        # CSV.write("csvs/rawcgs_$(d).csv", results)
    end
    close(reio)
    close(itio)
    return
end

get_resids()


;
