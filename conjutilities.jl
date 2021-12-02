# approximation algorithm from ``Algorithm 917: Complex Double-Precision
# Evaluation of the Wright ω Function'' by Piers W. Laurence, Robert M. Corless
# and David J. Jeffrey
# Series approximations can be found in the book chapter ``The Wright ω Function''
# by Corless, R. M. and Jeffrey, D. J.

# initial series approximations should be modified for types other than Float64 (implemented elsewhere)

function omegawright(z::T) where T
    z = float(z)
    if z < -2
        t = exp(z)
        w = t * (1 + t * (-1 + t * (T(3) / 2 + t * (-T(8) / 3 + T(125) / 24 * t))))
    elseif z < 1 + π
        z1 = z - 1
        w = 1 + z1 / 2 * (1 + z1 / 8 * (1 + z1 / 12 * (-1 + z1 / 16 *
            (-1 + z1 * T(13) / 20))))
    else
        lz = log(z)
        w = z + lz * (-1 + (1 + (lz / 2 - 1 + (lz * (lz / 3 - T(3) / 2) + 1) / z) / z) / z)
    end
    r = z - w - log(w)
    for k in 1:2
        t = (1 + w) * (1 + w + 2 / 3 * r)
        w = w * (1 + r / (1 + w) * (t - r / 2) / (t - r))
        r = (2 * w * (w - 4) - 1) / 72 / (1 + w)^6 * r^4
    end
    return w::float(T)
end

function rootnewton(
    f::Function,
    g::Function;
    lower::T = -Inf,
    upper::T = Inf,
    init::T = (lower + upper) / 2,
    increasing::Bool = true,
    ) where {T <: Real}
    curr = init
    f_new = f(BigFloat(curr))
    iter = 0
    while abs(f_new) > 1000eps()
        candidate = curr - f_new / g(BigFloat(curr))
        if (candidate < lower) || (candidate > upper)
            curr = (lower + upper) / 2
        else
            curr = candidate
        end
        f_new = f(BigFloat(curr))
        if f_new < 0
            if increasing
                lower = curr
            else
                upper = curr
            end
        else
            if increasing
                upper = curr
            else
                lower = curr
            end
        end
        iter += 1
        if iter > 200
            @warn "too many iters in dual grad"
            break
        end
    end
    # @show iter
    return (T(curr), iter)
end
