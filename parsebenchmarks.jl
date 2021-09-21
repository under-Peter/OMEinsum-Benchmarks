using JSON
using DataFrames
using Statistics, Dates
using BenchmarkTools

const emptydf = DataFrame((
           lang = String[],
           label = String[],
           dtime = DateTime[],
           vlang = String[],
           pcommit = Union{Missing,String}[],
           op = String[],
           ttype = String[],
           mtype = String[],
           memory = Union{Int,Missing}[],
           allocs = Union{Int,Missing}[],
           tmin    = Float64[],
           tmax    = Float64[],
           tmedian = Float64[],
           tmean   = Float64[],
           tstddev = Float64[]
           ))

parsepybenchjson(df, filename; label = "numpy") = parsepybenchjson!(deepcopy(df), filename; label = label)
function parsepybenchjson!(df::DataFrame, filename::String; label = "numpy")
    pjson = JSON.parsefile(filename)
    vlang = pjson["machine_info"]["python_version"]
    dtime = DateTime(pjson["datetime"][1:end-3])
    pcommit = missing
    foreach(pjson["benchmarks"]) do d
        name = d["name"]
        m = match(r"test_([a-z]*)\[([a-z]*)-([a-zA-Z0-9{}]*)\]", name)
        stats = d["stats"]
        op, mtype, ttype = string.(m.captures)
        push!(df, (
            lang = "python",
            label = label,
            dtime = dtime,
            vlang = vlang,
            pcommit = pcommit,
            op=op,
            ttype=ttype,
            mtype=mtype,
            memory = missing,
            allocs = missing,
            #python results are in seconds -> go to nanoseconds
            tmin    = 10^9*d["stats"]["min"],
            tmax    = 10^9*d["stats"]["max"],
            tmedian = 10^9*d["stats"]["median"],
            tmean   = 10^9*d["stats"]["mean"],
            tstddev = 10^9*d["stats"]["stddev"]
            ))
    end
    return df
end


parsejuliajson(df, filename; label = "") = parsejuliajson!(deepcopy(df), filename; label = label)
function parsejuliajson!(df::DataFrame, filename::String; label::String = "")
    jjson = JSON.parsefile(filename)
    vlang = match(r"Julia Version (\d*.\d*.\d*)",jjson["vinfo"]).captures[1]
    isnothing(vlang) && error("version couldn't be read")
    bmark = BenchmarkTools.load(IOBuffer(jjson["benchmarkgroup"]))[1]
    dtime = DateTime(jjson["date"])
    pcommit = jjson["commit"]

    foreach(leaves(bmark)) do (labels, trial)
        op, ttype, mtype = string.(labels)
        times = trial.times
        push!(df, (
            lang = "julia",
            label = label,
            dtime = dtime,
            vlang = vlang,
            pcommit = pcommit,
            op=op,
            ttype=ttype,
            mtype=mtype,
            memory = trial.memory,
            allocs = trial.allocs,
            tmin = minimum(times),
            tmax = maximum(times),
            tmedian = median(times),
            tmean = mean(times),
            tstddev = sqrt(var(times))
        ))
    end
    df
end

@doc raw"
    benchmarkscores(df::DataFrame, ref::String)
calculates a benchmarkscore for each label in `df` with the label `ref`
as a reference. The scores are calculated as:
``s_{\text{label}} = \Sum_{\text{ops},\text{sizes},\text{types}} w_\text{op} * log(t_\text{label}/t_\text{ref})``
where the weights can be found in the function definition.

Since missing terms are ignored, it is recommended to preparse the `df`.
"

function benchmarkscores(df, ref)
    weights = Dict{String,Float64}(
         "manyinds" => 1,
         "matmul" => 1,
         "batchmul" => 1,
         "dot" => 1,
         "trace" => 0.5,
         "ptrace" => 0.3,
         "diag" => 0.5,
         "perm" => 0.8,
         "tcontract" => 1,
         "star" => 0.7,
         "starandcontract" => 0.2,
         "indexsum" => 0.6,
         "hadamard" => 0.6,
         "outer" => 0.3)
    fdf = DataFrame(label = String[], score = Float64[])
    refdf = @where(df, :label .== ref)
    for x in groupby(df, :label)
        l = x.label[1]
        score = 0.
        for r in eachrow(x)
            op, ttype, mtype, t = r.op, r.ttype, r.mtype, r.tmin
            t0 = @where(refdf, :ttype .== ttype,
                               :op .== op,
                                :mtype .== mtype).tmin[]
            score += weights[op] * log(t/t0)
        end
        push!(fdf, (label = l, score = score))
    end
    fdf
end
