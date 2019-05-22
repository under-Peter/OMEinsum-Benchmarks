using JSON
using DataFrames
using Statistics, Dates
using BenchmarkTools

const emptydf = DataFrame((lang = String[],
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

parsepybenchjson(df, filename; which = "numpy") = parsepybenchjson!(deepcopy(df), filename, which = which)
function parsepybenchjson!(df::DataFrame, filename::String; which = "numpy")
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
            lang = "python-"*which,
            dtime = dtime,
            vlang = vlang,
            pcommit = pcommit,
            op=op,
            ttype=ttype,
            mtype=mtype,
            memory = missing,
            allocs = missing,
            #python esults are in seconds -> go to nanoseconds
            tmin    = 10^9*d["stats"]["min"],
            tmax    = 10^9*d["stats"]["max"],
            tmedian = 10^9*d["stats"]["median"],
            tmean   = 10^9*d["stats"]["mean"],
            tstddev = 10^9*d["stats"]["stddev"]
            ))
    end
    return df
end


parsejuliajson(df, filename) = parsejuliajson!(deepcopy(df), filename)
function parsejuliajson!(df::DataFrame, filename::String)
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
