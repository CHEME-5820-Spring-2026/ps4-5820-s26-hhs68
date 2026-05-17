# Hopfield storage capacity sweep for PS4 DQ1
include(joinpath(@__DIR__, "..", "Include.jl"))

function corrupt(v::Vector{Int64}, frac::Float64; seed::Int=42)::Vector{Int64}
    n = length(v)
    flip_idx = StatsBase.sample(MersenneTwister(seed), 1:n, round(Int, frac * n), replace=false)
    v_c = copy(v); v_c[flip_idx] .*= -1; return v_c
end

function tanimoto(a, b)
    d = dot(a, b); denom = sum(a) + sum(b) - d
    return denom == 0 ? 0.0 : d / denom
end

to_binary(v) = Int.(v .== 1)

function hopfield_recall(N_store, test_idx; digit=3, cf=0.30, steps=20, seed=2)
    np = 784
    n_load = max(N_store, test_idx) + 10
    dd = MyMNISTHandwrittenDigitImageDataset(number_of_examples=n_load)
    X = zeros(Int64, np, N_store)
    for i in 1:N_store
        b = Float64.(dd[digit][:, :, i])[:] .> 0.5
        X[:, i] = Int64.(2 .* b .- 1)
    end
    W = (X * X') ./ N_store
    W[diagind(W)] .= 0.0
    b = Float64.(dd[digit][:, :, test_idx])[:] .> 0.5
    v0 = Int64.(2 .* b .- 1)
    vc = corrupt(v0, cf; seed=seed)
    vs = copy(vc)
    for _ in 1:steps
        vs = sign.(W * vs)
        vs[vs .== 0] .= 1
    end
    tanimoto(to_binary(vs), to_binary(v0))
end

Ns = [30, 50, 80, 108, 120, 150, 200]
n_trials = 10

println("Notebook protocol (test idx = N+2), mean Tanimoto over $(n_trials) corruption seeds:")
for N in Ns
    ts = [hopfield_recall(N, N + 2; seed=s) for s in 1:n_trials]
    m = sum(ts) / length(ts)
    println("  N=$(lpad(N,3)): mean=$(round(m,digits=3))  range=[$(round(minimum(ts),digits=3)), $(round(maximum(ts),digits=3))]")
end

println("\nStored pattern 1 recall (classic), mean over $(n_trials) seeds:")
for N in Ns
    ts = [hopfield_recall(N, 1; seed=s) for s in 1:n_trials]
    m = sum(ts) / length(ts)
    println("  N=$(lpad(N,3)): mean=$(round(m,digits=3))  range=[$(round(minimum(ts),digits=3)), $(round(maximum(ts),digits=3))]")
end
