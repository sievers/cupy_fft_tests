import numpy as np
import cupy as cp
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
#sizes=sizes[:20];nchunk=len(sizes)

rsize=get_max_size(sizes)
csize=get_max_csize(sizes)
print('r/c sizes are ',rsize,csize)

#allocate buffers for our data.  we process one chunk at a time
#so we're happy to have all of our data chunks share memory buffers
ibuf=cp.random.randn(rsize,dtype='float32')
ftbuf=cp.empty(csize,dtype='complex64')
obuf=cp.empty(rsize,dtype='float32')


#set up slices so each chunk has its own view of a shared buffer
indat=[cp.reshape(ibuf[:sz[0]*sz[1]],[sz[0],sz[1]]) for sz in sizes]
ftdat=[cp.reshape(ftbuf[:sz[0]*(sz[1]//2+1)],[sz[0],sz[1]//2+1]) for sz in sizes]
outdat=[cp.reshape(obuf[:sz[0]*sz[1]],[sz[0],sz[1]]) for sz in sizes]

#t1=time.time()
#fcuplans=[None]*nchunk
#icuplans=[None]*nchunk

niter=5
for iter in range(niter):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    for i in range(nchunk):
        #for each problem size, we'll allocate an output array,
        #and create a plan via cufft.Plan1d.  We'll then call the fft
        #and delete the plan.  
        sz=sizes[i]
        tmp_ft=cp.empty([sz[0],sz[1]//2+1],dtype='complex64')
        fplan=cp.cuda.cufft.Plan1d(sz[1],cp.cuda.cufft.CUFFT_R2C,sz[0],out=ftdat[i])
        #fplan.fft(indat[i],ftdat[i],cp.cuda.cufft.CUFFT_FORWARD)
        fplan.fft(indat[i],tmp_ft,cp.cuda.cufft.CUFFT_FORWARD)
        del(fplan)
        tmp_out=cp.empty([sz[0],sz[1]],dtype='float32')
        iplan=cp.cuda.cufft.Plan1d(sz[1],cp.cuda.cufft.CUFFT_C2R,sz[0],out=outdat[i])
        iplan.fft(tmp_ft,tmp_out,cp.cuda.cufft.CUFFT_INVERSE)
        #iplan.fft(ftdat[i],outdat[i],cp.cuda.cufft.CUFFT_INVERSE)
        del(iplan)
        if (iter==niter-1) and(i==nchunk-1):
            outdat[i][:]=tmp_out
        del(tmp_out)
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('took ',t2-t1,' total seconds on iter ',iter,' with Plan1d for average time ',(t2-t1)/(nchunk*2),' per transform')

if False:
#sanity check to make sure we're getting correct answers
    aa=cp.asnumpy(indat[i])
    bb=cp.asnumpy(outdat[i])
    print('round trip fractional scatter is ',np.std(aa-bb/sz[1]))
    del(aa)
    del(bb)

cache = cp.fft.config.get_plan_cache()
cache.set_size(0)  #otherwise we run out of memory
for iter in range(5):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    for i in range(nchunk):
        tmpft=cp.fft.rfft(indat[i],axis=1)
        tmp_back=cp.fft.irfft(tmpft,n=indat[i].shape[1],axis=1)
        del(tmpft)
        del(tmp_back)
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('took ',t2-t1,' total seconds on iter ',iter,' with rfft/irfft for average time ',(t2-t1)/(nchunk*2),' per transform')

