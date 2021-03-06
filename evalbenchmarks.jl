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
benchmarkpkg("OMEinsum", "master", resultfile = "benchmarkfiles/juliabenchmarkmaster.json")
# benchmarkpkg("OMEinsum", "benchmark-einsumjl", resultfile = "benchmarkfiles/juliabenchmarkeinsumjl.json")
# isnothing(x) = x === nothing
# df = parsejuliajson(emptydf, "benchmarkfiles/juliabenchmarkeinsumjl.json", label = "einsumjl")
# df = parsepybenchjson(df, "benchmarkfiles/benchnumpy.json",label="numpy")
# df = parsepybenchjson(df, "benchmarkfiles/benchtorch.json",label="torch")
# df = parsepybenchjson(df, "benchmarkfiles/benchtorchgpu.json",label="torch-gpu")
# df = parsejuliajson(df, "benchmarkfiles/juliabenchmarkmaster.json",label="master")
# df = parsejuliajson(df, "benchmarkfiles/julibenchmarkgpu.json",label="julia-gpu")
# CSV.write("benchmarkdf.csv", df)

df = CSV.read("benchmarkdf.csv")

@df @where(df, :label .∈ Ref(("numpy","torch","master","einsumjl","julia-gpu","torch-gpu")),
               :ttype .== "Float32",
               :mtype .== "small") scatter(
    :op,
    :tmedian,
    ylims = (1, 10^13),
    group = :label,
    legend = :topleft,
    xrotation=35, xtickfont = font(8),
    yscale=:log10)

mtypes = ["small", "medium", "large"]
ttypes = ["Float64", "Float32", "Complex{Float64}", "Complex{Float32}"]
for mtype in mtypes
    for ttype in ttypes
        tmpdf = @where(df, :ttype .== ttype, :mtype .== mtype,
            :label .∈ Ref(("master", "numpy","torch","einsumjl","julia-gpu"#=,"torch-gpu"=#)),
            )
        tmax = maximum(tmpdf.tmin)
        p = @df tmpdf scatter(
            :op, :tmin, group = :label, yscale=:log10,
            legend = :top,
            marker = :auto,
            title = lowercase(ttype),
            ylabel = "ns",
            ylims = (1, tmax * 10^4),
            yticks = 10 .^ (0:13),
            xrotation=35, xtickfont = font(8))
        savefig(p, "plots/$(lowercase(ttype))-$(lowercase(mtype)).png")
    end
end


scoresdf = benchmarkscores(
    @where(df, :label .∈ Ref(("einsumjl","master","numpy", "torch","julia-gpu","torch-gpu")),
               :ttype .== "Float64"),
    "numpy")

CSV.write("scores.csv",scoresdf)
