using Plots
nvals=2 .^(6:9)
timings=[  0.172319  0.0014909
0.715925  0.0070021
3.12396   0.0297446
21.6781    0.135321]

xaxis=nvals
xaxist=string.(nvals)
yaxis=[0.001, 0.01, 0.1, 1,10, 60 ]
yaxist=["1 ms","10ms","0.1s","1s", "10s", "1min"]
plot(nvals, timings, labels= ["1" " 2"] , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3, linestyle=:dash)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(xaxis,xaxist), xlabel= "matrix size nxn", ylabel= "Execution time", title="title", dpi=1000)
