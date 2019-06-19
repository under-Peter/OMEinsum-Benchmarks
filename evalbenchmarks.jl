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
benchmarkpkg("OMEinsum", "benchmark", resultfile = "benchmarkfiles/juliabenchmark.json")
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

@df @where(df, :label .∈ Ref(("native","numpy","dispatch")),
               :ttype .== "Float64",
               :mtype .== "medium") scatter(
    :op,
    :tmin,
    ylims = (1, 10^13),
    group = :label,
    legend = :topleft,
    xrotation=35, xtickfont = font(8),
    yscale=:log10)

mtypes = ["tiny", "small", "medium", "large", "huge"]
for mtype in mtypes
    p = @df @where(df,
        :ttype .== "Float64",
        :mtype .== mtype,
        :label .∈ Ref(("dispatch", "native","numpy","torch","einsumjl")),
        ) scatter(
        :op, :tmin, group = :label, yscale=:log10,
        legend = :topleft,
        marker = :auto,
        ylabel = "ns",
        ylims = (1,10^13),
        yticks = 10 .^ (0:13),
        xrotation=35, xtickfont = font(8))
    savefig(p, "plots/float64-$(lowercase(mtype)).png")
end


scoresdf = benchmarkscores(
    @where(df, :label .∈ Ref(("einsumjl","dispatch","naive","numpy", "torch")),
               :ttype .== "Float64"),
    "numpy")
using CSV

CSV.write("scores.csv",scoresdf)
