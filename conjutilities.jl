# approximation algorithm from ``Algorithm 917: Complex Double-Precision
# Evaluation of the Wright ω Function'' by Piers W. Laurence, Robert M. Corless
# and David J. Jeffrey
# Series approximations can be found in the book chapter ``The Wright ω Function''
# by Corless, R. M. and Jeffrey, D. J.

function wrightomega(z::T) where {T <: Real}
    z = float(z)
    if z < -2
        t = exp(z)
        w = t * (1 + t * (-1 + t * (T(3) / 2 + t * (-T(8) / 3 + T(125) / 24 * t))))
    elseif z < T(1) + π
        z1 = z - 1
        w = 1 + z1 / 2 * (1 + z1 / 8 * (1 + z1 / 12 * (-1 + z1 / 16 *
            (-1 + z1 * T(13) / 20))))
    else
        lz = log(z)
        w = z + lz * (-1 + (1 + (lz / 2 - 1 + (lz * (lz / 3 - T(3) / 2) + 1) /
            z) / z) / z)
    end
    # refinement steps
    for _ in 1:5
        r = z - w - log(w)
        w1 = w + 1
        t = w1 * (w1 + T(2) / 3 * r)
        w *= 1 + r / w1 * (t - r / 2) / (t - r)
        fscn = abs(r^4 * (2 * w * (w - 4) - 1))
        if t < eps(float(T)) * 72 * w1^6
            break
        end
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
    f_new = f(curr)
    iter = 0
    while abs(f_new) > 1000eps()
        candidate = curr - f_new / g(BigFloat(curr))
        if (candidate < lower) || (candidate > upper)
            curr = (lower + upper) / 2
        else
            curr = candidate
        end
        f_new = f(curr)
        if (f_new < 0 && increasing) || (f_new >= 0 && !increasing)
            lower = curr
        else
            upper = curr
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
