import numpy as np
import cupy as cp
import pycufft
import time


def get_max_size(sizes):
    #find the maximum array size, in elements, for a list of sizes
    return np.max([sz[0]*sz[1] for sz in sizes])

def get_max_csize(sizes):
    #find the maximum output r2c size for a list of real sizes
    #size is in complex elements, so would be about half of get_max_size
    return np.max([sz[0]*((sz[1]//2+1)) for sz in sizes])

#these are a bunch of similar fft lengths that are the product of small prime factors
#lens=np.asarray([250880, 252000, 252105, 253125, 254016, 255150, 256000, 257250, 258048, 259200, 259308, 262144, 262440, 262500, 263424, 264600, 268800, 268912, 270000, 272160])
lens=np.asarray([250880, 252000, 254016, 255150, 256000, 257250, 258048, 259200, 259308, 262144, 262440, 262500, 263424, 264600, 268800, 268912, 270000, 272160])
#pick a set of random sizes for the other axis
ndet=np.asarray([600+np.linspace(0,200,10)],dtype='int')


#now set up sizes for the ffts we're going to carry out.  For this test case,
#we'll pair each of the lens with each of the ndets, but in practice
#these numbers would be random-ish, depending on how our detectors were
#feeling that day.
n,m=np.meshgrid(ndet,lens)
n=np.ravel(n)
m=np.ravel(m)
nchunk=len(n)  #how many chunks of data we're going to be processing
sizes=[[n[i],m[i]] for i in range(nchunk)]

plans=pycufft.MultiPlan(sizes)
rsize=get_max_size(sizes)
csize=get_max_csize(sizes)


#allocate buffers for our data.  we process one chunk at a time
#so we're happy to have all of our data chunks share memory buffers
ibuf=cp.ones(rsize,dtype='float32')
ftbuf=cp.empty(csize,dtype='complex64')
obuf=cp.empty(rsize,dtype='float32')

#set up slices so each chunk has its own view of a shared buffer
indat=[cp.reshape(ibuf[:sz[0]*sz[1]],[sz[0],sz[1]]) for sz in sizes]
ftdat=[cp.reshape(ftbuf[:sz[0]*(sz[1]//2+1)],[sz[0],sz[1]//2+1]) for sz in sizes]
outdat=[cp.reshape(obuf[:sz[0]*sz[1]],[sz[0],sz[1]]) for sz in sizes]


for iter in range(5):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    for i in range(nchunk):
        pycufft.rfft(indat[i],ftdat[i],plan_cache=plans)
        pycufft.irfft(ftdat[i],outdat[i],sizes[i][1],plan_cache=plans)
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('took ',t2-t1,' total seconds on iter ',iter,' for average time ',(t2-t1)/(nchunk*2),' per transform')


ft=cp.fft.rfft(indat[0],axis=1) #do a first fft just to make sure it's warmed up
for iter in range(5):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    for i in range(nchunk):
        ft=cp.fft.rfft(indat[i],axis=1)
        out=cp.fft.irfft(ft,axis=1,n=sizes[i][1])
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('naive cupy took ',t2-t1,' seconds on iter ',iter,', for average time ',(t2-t1)/(nchunk*2),' per transform')

cache = cp.fft.config.get_plan_cache()
for iter in range(5):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    for i in range(nchunk):
        ft=cp.fft.rfft(indat[i],axis=1)
        out=cp.fft.irfft(ft,axis=1,n=sizes[i][1])
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('planned cupy took ',t2-t1,' seconds on iter ',iter,', for average time ',(t2-t1)/(nchunk*2),' per transform')
