using Plots, StatsPlots
using DataFramesMeta
using CSV

#=
 run julia-benchmarks
=#
# include("parsebenchmarks.jl")
# using PkgBenchmark
# using OMEinsum
# benchmarkpkg("OMEinsum", "benchmark", resultfile = "benchmarkfiles/juliabenchmark.json")
# df = parsejuliajson(emptydf, "benchmarkfiles/juliabenchmark.json")
# df = parsepybenchjson(df, "benchmarkfiles/benchnumpy.json",which="numpy")
# df = parsepybenchjson(df, "benchmarkfiles/benchtorch.json",which="torch")
# CSV.write("benchmarkdf.csv", df)

df = CSV.read("benchmarkdf.csv")

mtypes = ["small", "medium", "large", "huge"]
for mtype in mtypes
    p = @df @where(df, :ttype .== "Float64", :mtype .== mtype) scatter(
        :op, :tmin, group = :lang, yscale=:log10,
        legend = :topleft,
        marker = [:diamond :circle :square],
        ylabel = "ns",
        xrotation=35, xtickfont = font(8))
    savefig(p, "plots/float64-$(lowercase(mtype)).png")
end
