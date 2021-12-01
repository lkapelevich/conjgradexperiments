import Random
import ForwardDiff
import DiffResults
using LinearAlgebra
using DataFrames
using CSV
using Statistics
include("conjutilities.jl")
# include("grad_hess.jl")
Random.seed!(1)
num_samples = 1
cone_d = 50
offsets = 10.0 .^ collect(-5:-1)

##
abstract type Cone end
struct SumLog <: Cone end
struct HypoPower <: Cone
    α::Vector{Float64}
end
HypoPower() = (α = rand(cone_d); α /= sum(α); HypoPower(α))
struct HypoGeom <: Cone end
struct RadialPower <: Cone
    α::Vector{Float64}
end
RadialPower() = (α = rand(cone_d); α /= sum(α); RadialPower(α))
struct RadialGeom <: Cone end
struct InfinityNorm <: Cone end

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
function barrier(::HypoGeom)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return -log(exp(sum(log(wi) / cone_d for wi in w)) - u) - sum(log, w)
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
function barrier(::RadialGeom)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return -log(exp(sum(2 / cone_d * log(wi) for wi in w)) - u^2) -
            sum((cone_d - 1) / cone_d * log(wi) for wi in w)
    end
    return bar
end
function barrier(::InfinityNorm)
    function bar(uw)
        (u, w) = (uw[1], uw[2:end])
        return sum(-log(u^2 - wi^2) for wi in w) + (cone_d - 1) * log(u)
    end
    return bar
end

init_guess(::SumLog) = vcat(-1, 1, ones(cone_d))
init_guess(::Union{HypoPower, HypoGeom}) = vcat(-1, ones(cone_d))
init_guess(cone::RadialPower) = vcat(0, sqrt.(1 .+ cone.α))
init_guess(::RadialGeom) = vcat(0, fill(sqrt.(1 + inv(cone_d)), cone_d))
init_guess(::InfinityNorm) = vcat(1, zeros(cone_d))

function dual_point(::SumLog, offset::Float64)
    r = rand(cone_d)
    p = -rand()
    phi = (p * sum(log(-ri / p) for ri in r) + p * cone_d)
    q = phi * (1 + sign(phi) * offset)
    return vcat(p, q, r)
end
function dual_point(cone::HypoPower, offset::Float64)
    α = cone.α
    r = rand(cone_d)
    phi = -exp(sum(αi * log(ri / αi) for (ri, αi) in zip(r, α)))
    p = phi * (1 - offset)
    return vcat(p, r)
end
dual_point(::HypoGeom, offset::Float64) = dual_point(HypoPower(), offset)
function dual_point(cone::RadialPower, offset::Float64)
    α = cone.α
    r = rand(cone_d)
    # only look at the p > 0 case, but that's OK because residuals aren't
    # going to behave differently in the other direction
    phi = exp(sum(αi * log(ri / αi) for (ri, αi) in zip(r, α)))
    p = phi * (1 - offset)
    return vcat(p, r)
end
dual_point(::RadialGeom, offset::Float64) = dual_point(RadialPower(), offset)
function dual_point(::InfinityNorm, offset::Float64)
    r = randn(cone_d)
    p = sum(abs, r) * (1 + offset)
    return vcat(p, r)
end

function grad_hess(cone::Cone, s)
    bar = barrier(cone)
    # s = BigFloat.(s)
    result = DiffResults.HessianResult(s)
    return ForwardDiff.hessian!(result, bar, s);
end

function dual_grad(::SumLog, pqr)
    (p, q, r) = (pqr[1], pqr[2], pqr[3:end])
    dual_ϕ = sum(log(r_i / -p) for r_i in r)
    β = 1 + cone_d - q / p + dual_ϕ
    bomega = cone_d * omegawright(β / cone_d - log(cone_d))
    @assert bomega + cone_d * log(bomega) ≈ β

    cgp = (-cone_d - 2 + q / p + 2 * bomega) / (p * (1 - bomega))
    cgq = -1 / (p * (1 - bomega))
    cgr = bomega ./ r / (1 - bomega)
    return (vcat(cgp, cgq, cgr), 0)
end
function dual_grad(cone::HypoPower, pr)
    (p, r) = (pr[1], pr[2:end])
    α = cone.α
    sumlog = sum(α_i * log(r_i) for (α_i, r_i) in zip(α, r))

    f(y) = sum(ai * log(y - p * ai) for ai in α) - sumlog
    fp(y) = sum(ai / (y - p * ai) for ai in α)
    lower_bound = 0.0
    upper_bound = exp(sumlog) + p / cone_d
    (new_bound, iter) = rootnewton(lower_bound, upper_bound, f, fp)

    dual_g_ϕ = inv(new_bound)
    cgp = -inv(p) - dual_g_ϕ
    cgr = (p * dual_g_ϕ * α .- 1) ./ r
    return (vcat(cgp, cgr), iter)
end
function dual_grad(cone::HypoGeom, pr)
    (p, r) = (pr[1], pr[2:end])
    dual_ϕ = exp(sum(log, r) / cone_d)
    cgp = -inv(p) - cone_d / (cone_d * dual_ϕ + p)
    cgr = -1 / (1 + p / cone_d / dual_ϕ) ./ r
    return (vcat(cgp, cgr), 0)
end
function dual_grad(cone::RadialPower, pr)
    (p, r) = (pr[1], pr[2:end])
    α = cone.α
    if iszero(p)
        ctp = 0
        cgr = -(α .+ 1) ./ r
    else
        log_phi_r = 2 * sum(αi * log(ri) for (αi, ri) in zip(α, r))
        phi_r = exp(log_phi_r)
        phi_αr = exp(sum(αi * log(ri / αi) for (αi, ri) in zip(α, r)))
        inner_bound = -1 / p - (1 + sign(p) * 1 / p * sqrt(phi_r * (cone_d^2 /
            p^2 * phi_r + cone_d^2 - 1))) / (p / cone_d - cone_d * phi_r / p)
        gamma = abs(p) / phi_αr
        outer_bound = (1 + cone_d) * gamma / (1 - gamma) / p
        f(y) = 2 * sum(αi * log(2 * αi * y^2 + (1 + αi) * 2 * y / p) for αi in α) -
            log_phi_r - log(2 * y / p + y^2) - 2 * log(2 * y / p)
        fp(y) = 2 * sum(αi^2 / (αi * y + (1 + αi) / p) for αi in α) -
            2 * (y + 1 / p) / y / (y + 2 / p)
        (cgp, iter) = rootnewton(outer_bound, inner_bound, f, fp, outer_bound)
        cgr = -(α * (1 + p * cgp) .+ 1) ./ r
    end
    return (vcat(cgp, cgr), iter)
end
function dual_grad(cone::RadialGeom, pr)
    (p, r) = (pr[1], pr[2:end])
    if iszero(p)
        ctp = 0
        cgr = -(inv(cone_d) .+ 1) ./ r
    else
        phi_r = exp(2 * sum(log(ri) / cone_d for ri in r))
        cgp = -1 / p - (1 + sign(p) * 1 / p * sqrt(phi_r * (cone_d^2 /
            p^2 * phi_r + cone_d^2 - 1))) / (p / cone_d - cone_d * phi_r / p)
        cgr = -((1 + p * cgp) / cone_d + 1) ./ r
    end
    return (vcat(cgp, cgr), 0)
end
function dual_grad(cone::InfinityNorm, pr)
    (p, r) = (pr[1], pr[2:end])
    dual_zeta = p - sum(abs, r)
    h(y) = p * y + sum(sqrt(1 + abs2(ri) * y^2) for ri in r) + 1
    hp(y) = p + sum(abs2(ri) * y * (1 + abs2(ri) * y^2)^(-1/2) for ri in r)
    lower = -(cone_d + 1) / dual_zeta
    upper = min(-inv(dual_zeta), -(cone_d + 1) / p)
    (cgp, iter) = rootnewton(lower, upper, h, hp)

    cgr = copy(r)
    for i in eachindex(r)
        if abs(r[i]) .< 100eps()
            cgr[i] = cgp^2 * r[i] / 2
        else
            cgr[i] = (-1 + sqrt(1 + abs2(r[i]) * cgp^2)) / r[i]
        end
    end
    return (vcat(cgp, cgr), iter)
end

function naive_dual_grad(cone, z)
    curr = init_guess(cone)
    derivs = grad_hess(cone, curr)
    (g, H) = (DiffResults.gradient(derivs), DiffResults.hessian(derivs))
    r = z + g
    Hir = H \ r
    # (r, Hir) = grad_invhess(cone, curr, z)
    n = sqrt(dot(r, Hir))

    max_iter = 400
    iter = 1
    resid_path = []

    while n > 1000eps()
        push!(resid_path, dot(z, -curr) + cone_d + 1)
        α = (n > 0.35 ? inv(1 + n) : 1)
        curr -= Hir * α
        derivs = grad_hess(cone, curr)
        (g, H) = (DiffResults.gradient(derivs), DiffResults.hessian(derivs))
        r = z + g    
        Hir = H \ r
        # (r, Hir) = grad_invhess(cone, curr, z)
        n2 = dot(r, Hir)
        n2 < 0 && break # TODO check magnitude
        n = sqrt(n2)
        iter += 1
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

function get_resids()
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
        cone = C()
        z = dual_point(cone, o)
        (g, iter) = dual_grad(cone, z)
        resid = dot(z, g) + length(z)
        # @show resid
        (naive_g, naive_iter, resid_path) = naive_dual_grad(cone, z)
        naive_resid = dot(z, naive_g) + length(z)
        push!(results, (cone_d, nameof(C), o, i, resid,
            naive_resid, naive_iter, iter))
        if i == 1
            resid_df = DataFrame(resid = resid_path)
            resids_name = lowercase(string(nameof(C))) * string(Int(-log10(o)))
            CSV.write(resids_name * ".csv", resid_df)
        end
    end

    agg = combine(groupby(results, [:cone_d, :cone_type, :offset]),
        :residdirect => mean => :residdirectmean,
        :residnaive => mean => :residnaivemean,
        :newtoniters => mean => :newtonitersmean,
        :directiters => mean => :directitersmean,
        )

    CSV.write("cgs.csv", agg)
    return
end

get_resids()


;
