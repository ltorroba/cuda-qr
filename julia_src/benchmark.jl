using KernelAbstractions,CUDA,Random, LinearAlgebra, Printf, GPUArrays
backend=KernelAbstractions.get_backend(CUDA.zeros(2))
include("qr_kernels.jl")
include("cusol_funcs.jl")


elty=Float32
sizes=[32,128,512,1024,2048,4096]

println( " size    type   RRMSE    time (ms)  vs CUSOLVER  cutime(ms)  cutime_ext  type     RRMSE     time (ms)  vs CUSOLVER    cutime(ms)  cutime_ext");
println(" ------  ----  --------  --------  ------------  ----------  ----------  ------  ---------  --------   -------------  ----------  -----------");
for (i,size_i) in enumerate(sizes)
    nbtiles=Int(size_i/32)
    A=CUDA.randn( elty,size_i, size_i)
    C=CUDA.randn( elty,size_i, size_i)
    Tau=CUDA.zeros(nbtiles,size_i)
    Tau2=CUDA.zeros(size_i)
    dh = CUSOLVER.dense_handle()
    buffersize= geqrf_buffersize(A)
    buffer=CUDA.zeros(elty,buffersize)

    mybelapsed(geqrf!,A, Tau2, size_i, size_i, size_i,dh,buffer, buffersize)
    mybelapsed(CUSOLVER.geqrf!,A, Tau2)
    mybelapsed(mygeqrf!,A, Tau, nbtiles)
    t_cusolver = mybelapsed(geqrf!,A, Tau2, size_i, size_i, size_i,dh,buffer, buffersize)
    t_cusolverfull = mybelapsed(CUSOLVER.geqrf!,A, Tau2)
    t_KA = mybelapsed(mygeqrf!,A, Tau, nbtiles)
    
    Acpy=copy(A)
    Acpy2=copy(A)
    match = norm(triu!(abs.(mygeqrf!(Acpy, Tau, nbtiles)) - abs.(geqrf!(Acpy2, Tau2, size_i, size_i, size_i,dh,buffer, buffersize))))/norm(geqrf!(A, Tau2, size_i, size_i, size_i,dh,buffer, buffersize))
    @printf " %4d     QR    %8.02e    %7.02f  %8.02f %% %10.02f %10.2f" size_i match t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 1000*t_cusolverfull

    match = norm((abs.(myormqr!(copy(C), Acpy, Tau, nbtiles)) - abs.(CUSOLormqr!(copy(C),Acpy2,Tau2))))/norm(CUSOLormqr!(copy(C),Acpy2,Tau2))
    buffersize= ormqr_bufferSize( A, Tau2, C)
    buffer=CUDA.zeros(buffersize)

    mybelapsed(ormqr!,C, A, Tau2, dh, size_i, size_i, size_i,size_i,size_i,buffersize,buffer)
    mybelapsed(CUSOLormqr!,C,A,Tau2)
    mybelapsed(myormqr!,C, A, Tau, nbtiles)
    t_cusolver = mybelapsed(ormqr!,C, A, Tau2, dh, size_i, size_i, size_i,size_i,size_i,buffersize,buffer)
    t_cusolverfull = mybelapsed(CUSOLormqr!,C,A,Tau2)
    t_KA = mybelapsed(myormqr!,C, A, Tau, nbtiles)
    @printf "      Qmul  %8.02e   %7.02f  %8.02f %%    %10.02f %10.2f \n" match t_KA*1000  t_cusolver/t_KA*100 1000*t_cusolver 1000*t_cusolverfull
    
end  



#to do: @cuda always_inline=true,  @cuda fastmath=true,int32, streams
