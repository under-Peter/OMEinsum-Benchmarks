using Plots, StatsPlots
using DataFramesMeta
using DataFrames
using CSV

#=
 run julia-benchmarks
=#
include("parsebenchmarks.jl")
using PkgBenchmark
using OMEinsum
# benchmarkpkg("OMEinsum", "benchmark-einsumjl", resultfile = "benchmarkfiles/juliabenchmarkeinsumjl.json")
# benchmarkpkg("OMEinsum", "benchmark-einsumjl", resultfile = "benchmarkfiles/juliabenchmarkeinsumjl.json")
# benchmarkpkg("OMEinsum", "naive-einsum-bm", resultfile = "benchmarkfiles/juliabenchmarknaiv.json")
# benchmarkpkg("OMEinsum", "dispatch-testing", resultfile = "benchmarkfiles/juliabenchdispatch.json")
# benchmarkpkg("OMEinsum", "benchmark-tensoroperations", resultfile = "benchmarkfiles/juliabenchmarktensoroperations.json")
# df = parsejuliajson(emptydf, "benchmarkfiles/juliabenchmarkeinsumjl.json", label = "einsumjl")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmarktensoroperations.json", label="tensorops")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmark.json", label="native")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmark2.json", label="exp-space")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmarknaiv.json", label="naive")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchdispatch.json", label="dispatch")
# df = parsepybenchjson(df, "benchmarkfiles/benchnumpy.json",label="numpy")
# df = parsepybenchjson(df, "benchmarkfiles/benchtorch.json",label="torch")
# CSV.write("benchmarkdf.csv", df)

df = CSV.read("benchmarkdf.csv")

gr()
tmpdf = @where(df,
    :label .∈ Ref(("exp-space","naive", "dispatch","native","numpy")),
    :mtype .== "tiny",
    :ttype .== "Float64"
    )
p = @df tmpdf scatter(
    :op, :tmin, group = (:label), yscale=:log10,
    # legend = nothing,
    # legend = :topleft,
    # marker = :auto,
    ylims = (1, 10^13),
    yticks = 10 .^ (0:13),
    ylabel = "ns",
    xrotation=35, xtickfont = font(8))
savefig("test")

mtypes = ["tiny", "small", "medium", "large", "huge"]
for mtype in mtypes
    p = @df @where(df, :ttype .== "Float64", :mtype .== mtype) scatter(
        :op, :tmin, group = :label, yscale=:log10,
        legend = :topleft,
        marker = :auto,
        ylabel = "ns",
        ylims = (1,10^13),
        yticks = 10 .^ (0:13),
        xrotation=35, xtickfont = font(8))
    savefig(p, "plots/float64-$(lowercase(mtype)).png")
end


function benchmarkscores(df, ref)
    weights = Dict{String,Float64}(
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
            op = r.op
            ttype = r.ttype
            mtype = r.mtype
            t0 = @where(refdf, :ttype .== ttype, :op .== op, :mtype .== mtype).tmin[]
            t = r.tmin
            score += weights[op] * log(t/t0)
        end
        push!(fdf, (label = l, score = score))
    end
    fdf
end

benchmarkscores(df, "numpy")
