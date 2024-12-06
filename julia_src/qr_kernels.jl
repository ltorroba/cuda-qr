using KernelAbstractions.Extras: @unroll

@kernel function QR_unsafe_kernel_2d!(input, tau) 
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (N + 1, N)
    cache = @localmem eltype(input) (N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[i, j] = input[i, j]


    for iter in 1:N-1
        if (i > iter) && (j == iter)
            cache[i] = tile[i, iter]^2
        end
        @synchronize
        if (i == 1) && (j == 1)
            tmp_sum = zero(eltype(input))
            for l in iter+1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter, iter]) * tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        if (j >= iter) && (i >= iter)
            tmp_sum = zero(eltype(input))
            for k = iter+1:N
                tmp_sum += tile[k, iter]  * tile[k, j]
            end
        end
        tileiterj=tile[iter, j]
        tileiiter = tile[i, iter] 
        @synchronize
        if (j >= iter) && (i >= iter)
            corrvalue1 = corrvalue[1]
            tmp_sum = (tmp_sum / corrvalue1+ tileiterj)*tau_iter[1] 
            tileiiter = tileiiter / corrvalue1

            if (j==iter) && (i > iter) 
                tile[i, j] = tileiiter 
            elseif (i>iter)
                tile[i, j] = tile[i, j] - tileiiter* tmp_sum  
            else
                tile[i, j] = tile[i, j] - tmp_sum 
            end
        end
        @synchronize
    end
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end


@kernel function QR_unsafe_kernel2_2d!(input, input2, tau)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(input) (2N + 1, N)
    cache = @localmem eltype(input) (2N + 1)
    tau_iter = @localmem eltype(input) (1)
    corrvalue = @localmem eltype(input) (1)

    @inbounds tile[N+i, j] = input2[i, j]
    @inbounds tile[i, j] = input[i, j]
    
    @synchronize
    for iter in 1:N
        if (j==iter)
            cache[i] = tile[i+N, iter]^2
        end
        @synchronize
        if (i == 1) && (j==1)
            tmp_sum = zero(eltype(input))
            for l in 1:N
                tmp_sum += cache[l]
            end
            tmp_sum2 = sqrt(tmp_sum + tile[iter, iter]^2)
            newvalue = tile[iter, iter] + sign(tile[iter,iter]) *tmp_sum2
            tmp_sum2 = sqrt(tmp_sum + newvalue^2)
            tau_iter[1] = 2 * (newvalue / tmp_sum2)^2
            corrvalue[1] = newvalue
            tau[iter] = tau_iter[1]
        end
        tileiNiter= tile[i+N, iter]
        tileiterj=tile[iter, j]
        if (j>=iter)
            tmp_sum = zero(eltype(input))
            for l = N+1:2N
                tmp_sum += tile[l, iter] * tile[l, j]
            end
        end
        @synchronize
        taucorr=tau_iter[1] / corrvalue[1]
        corrvalue1 = corrvalue[1]
        if (j >= iter) 
            tmp_sum += corrvalue1 * tileiterj
            if (i==iter)
                tile[i, j] = tile[i,j] - tmp_sum * taucorr
            end
            if (j>iter)
                tile[i+N, j] = tile[i+N, j] - tileiNiter * tmp_sum *taucorr / corrvalue1
            end
        end
        if (j==1)
            tile[i+N, iter] = tileiNiter / corrvalue1
        end
        @synchronize
    end
    
    @inbounds input2[i, j] = tile[N+i, j]
    @inbounds input[i, j] = tile[i, j]
    @synchronize

end

@kernel function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau))
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N + 1,K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange = (1:N-1) 
    
    @synchronize
    for k in applyrange
        tmp_sum = zero(eltype(A))
        for l in k+j:K:N
            tmp_sum += M[l, k] * tile[l, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i] 
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        for l in k+j:K:N
            tile[l, i] = tile[l, i] - tmp_sum * M[l, k]
        end
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N]  = tile[i, l]
    end
end

@kernel function applyQorQt_unsafe_kernel2_2d!(A, B, @Const(Min), @Const(tau))
    g, _ = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]
    K = @uniform @groupsize()[2]
    tile = @localmem eltype(A) (2N + 1, N)
    M = @localmem eltype(A) (N + 1, N)
    cache = @localmem eltype(A) (N+1, K)
    
    @unroll for l in j:K:N
        @inbounds tile[i, l] = A[i, l+(g-1)*N]
        @inbounds tile[i+N, l] = B[i, l+(g-1)*N]
        @inbounds M[i, l] = Min[i, l]
    end

    applyrange =  (1:N) 

    @synchronize
    for k in applyrange
        tmp_sum= zero(eltype(A))       
        for j in j:K:N
            tmp_sum += M[j, k] * tile[j+N, i]
        end
        cache[i,j]=tmp_sum
        @synchronize
        tmp_sum = tile[k, i]
        for l in 1:K
            tmp_sum+=cache[i,l]
        end
        tmp_sum = tmp_sum * tau[k]
        if (j==1)
            tile[k, i] = tile[k, i] - tmp_sum
        end
        for l in j:K:N
            tile[l+N, i] = tile[l+N, i] - tmp_sum * M[l, k]
        end
        @synchronize
    end
    @unroll for l in j:K:N
        @inbounds A[i, l+(g-1)*N] = tile[i, l]
        @inbounds B[i, l+(g-1)*N] = tile[i+N, l]
    end
end



TILE_SIZE=32
TILE_SIZE2=4 

get_tileview(A, row , col, TILE_SIZEx, TILE_SIZEy ) = view(A, (row-1)*TILE_SIZEx.+(1:TILE_SIZEx),(col-1)*TILE_SIZEy.+(1:TILE_SIZEy))
get_rowview(A, row, startcol, TILE_SIZEx, TILE_SIZEy) =  view(A, (row-1)*TILE_SIZEx .+(1:TILE_SIZEx),((startcol-1)*TILE_SIZEy +1):size(A,2))
get_kernel_dims(::KernelAbstractions.Kernel{B,S}) where {B,S} = S.parameters[1]

QR1!(A, Tau, k) = QR_unsafe_kernel_2d!(backend, (TILE_SIZE,TILE_SIZE))( get_tileview(A, k,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, k,k, 1, TILE_SIZE), ndrange=(TILE_SIZE,TILE_SIZE)) 
QR2!(A, Tau, row, k) =QR_unsafe_kernel2_2d!(backend, (TILE_SIZE,TILE_SIZE))(get_tileview(A, k,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(A, row,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, row,k, 1, TILE_SIZE), ndrange=(TILE_SIZE,TILE_SIZE))

Qtapply1_par!(A, Tau, k) = applyQorQt_unsafe_kernel_2d!(backend, (TILE_SIZE,TILE_SIZE2))(get_rowview(A, k, k+1, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(A, k,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, k,k, 1, TILE_SIZE), ndrange=( size(A,2)-k*TILE_SIZE,TILE_SIZE2) )
Qtapply2_par!(A, Tau, row,k) = applyQorQt_unsafe_kernel2_2d!(backend, (TILE_SIZE,TILE_SIZE2))(get_rowview(A, k, k+1, TILE_SIZE, TILE_SIZE), 
                                    get_rowview(A, row, k+1, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(A, row,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, row,k, 1, TILE_SIZE), ndrange=( size(A,2)-k*TILE_SIZE,TILE_SIZE2))

Qtapply1_par_full!(B, A, Tau, k) = applyQorQt_unsafe_kernel_2d!(backend, (TILE_SIZE,TILE_SIZE2))(view(B, (1:TILE_SIZE).+(k-1)*TILE_SIZE,: ), 
                                    get_tileview(A, k,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, k,k, 1, TILE_SIZE), ndrange=( size(B,2),TILE_SIZE2) )
Qtapply2_par_full!(B, A, Tau, row,k) = applyQorQt_unsafe_kernel2_2d!(backend, (TILE_SIZE,TILE_SIZE2))(view(B, (1:TILE_SIZE).+(k-1)*TILE_SIZE,: ), 
                                    view(B, (1:TILE_SIZE).+(row-1)*TILE_SIZE,: ), 
                                    get_tileview(A, row,k, TILE_SIZE, TILE_SIZE), 
                                    get_tileview(Tau, row,k, 1, TILE_SIZE), ndrange=( size(B,2),TILE_SIZE2))

#Threads.@spawn begin, CUDA.@sync begin
    
function mygeqrf!(A, Tau, nbtiles)
    for k in 1:(nbtiles-1)
        QR1!(A,Tau, k)
        Qtapply1_par!(A, Tau, k)
        for row in k+1:nbtiles
            QR2!(A,Tau, row,k)
            Qtapply2_par!(A,Tau, row,k)
        end
    end
    QR1!(A,Tau, nbtiles)
    return A
end


function myormqr!(B, A, Tau, nbtiles)
    for k in 1:(nbtiles)
        Qtapply1_par_full!(B,A, Tau, k)
        for row in k+1:nbtiles
            Qtapply2_par_full!(B,A,Tau, row,k)
        end
    end
    return B
end