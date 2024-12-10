using Plots, StatsPlots
nvals=[128,512,1024,2048,4096,8096]
timings=[ 0.81	0.57	0.14	0.11	0.01
16.66	5.14	2.82	2.21	1.9
71.46	20.91	12.17	9.47	8.86
291.2	86.32	49.61	39.11	37.97
1252.07	353.07	202.89	159.11	156.92
7361.48	1346.71	803.63	642.47	638.48
]

#timings./=sum(timings,dims=2)
xaxis=nvals
xaxist=string.(nvals)
yaxis=[0.01,0.1,1, 10, 100, 1000 ]
yaxist=["0.01ms","0.1ms","1 ms","10ms","0.1s","1s"]
plot(nvals, timings, labels= ["kernel launches" "QR1" "QR2" "Qmul1" "Qmul2"] , lw=2, markershape=:circle, markerstrokewidth=0, markersize=3)#, linestyle=:dash)
plot!(xaxis=:log2,  yaxis=:log10,legend=:topleft,  yticks=(yaxis,yaxist), xticks=(xaxis,xaxist), xlabel= "matrix size nxn", ylabel= "Execution time", title="", dpi=1000)
savefig("reltime.png")

#groupedbar(string.(nvals), timings, bar_position = :stack, yticks=([0,0.5,1],["0%","50%","100%"]), bar_width=0.7,labels= ["kernel launches" "QR1" "QR2" "Qmul1" "Qmul2"] , xlabel= "matrix size nxn", ylabel= "Execution time")
