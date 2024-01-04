import numpy as np
import cupy as cp
import time


cufft_type = cp.cuda.cufft.CUFFT_R2C
n=2**18
batch=768
dx=cp.ones([batch,n],dtype='float32')
dx2=cp.empty([batch,n],dtype='float32')
out=cp.empty([batch,n//2+1],dtype='complex64')

cache = cp.fft.config.get_plan_cache()
mempool=cp.get_default_memory_pool()
print('usage is ',mempool.used_bytes()/1e6,' MB before making plans.')
fplan=cp.cuda.cufft.Plan1d(n,cp.cuda.cufft.CUFFT_R2C,batch,out=out)
iplan=cp.cuda.cufft.Plan1d(n,cp.cuda.cufft.CUFFT_C2R,batch,out=dx2)
print('usage is ',mempool.used_bytes()/1e6,' MB after making plans.')

cache.show_info()

fplan.fft(dx,out,cp.cuda.cufft.CUFFT_FORWARD)
iplan.fft(out,dx2,cp.cuda.cufft.CUFFT_FORWARD)


for i in range(0):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    fplan.fft(dx,out,cp.cuda.cufft.CUFFT_FORWARD)
    iplan.fft(out,dx2,cp.cuda.cufft.CUFFT_FORWARD)
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('average time ',(t2-t1)/2)
