using Plots, StatsPlots
using DataFramesMeta
using CSV

#=
 run julia-benchmarks
=#
# include("parsebenchmarks.jl")
# using PkgBenchmark
# using OMEinsum
# benchmarkpkg("OMEinsum", "benchmark-einsumjl", resultfile = "benchmarkfiles/juliabenchmarkeinsumjl.json")
# benchmarkpkg("OMEinsum", "benchmark-tensoroperations", resultfile = "benchmarkfiles/juliabenchmarktensoroperations.json")
# df = parsejuliajson(emptydf, "benchmarkfiles/juliabenchmarkeinsumjl.json", label = "einsumjl")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmarktensoroperations.json", label="tensorops")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmark.json", label="native")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmark2.json", label="exp-space")
# df = parsepybenchjson(df, "benchmarkfiles/benchnumpy.json",label="numpy")
# df = parsepybenchjson(df, "benchmarkfiles/benchtorch.json",label="torch")
# CSV.write("benchmarkdf.csv", df)

df = CSV.read("benchmarkdf.csv")

p = @df @where(df, :ttype .== "Float64", :mtype .== "medium") scatter(
    :op, :tmin, group = (:label), yscale=:log10,
    legend = :topleft,
    marker = :auto,
    ylims = (1, 10^13),
    yticks = 10 .^ (0:13),
    ylabel = "ns",
    xrotation=35, xtickfont = font(8))

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
