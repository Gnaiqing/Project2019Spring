import tvm,random,math,time
import numpy as np

def schedule_gemm_with(ops, bufs, bn, rn, bm):
    s = tvm.create_schedule(ops)
    s.cache_write(bufs[2], 'global')

    A, B, CC, C = s.stages

    xo, yo, xi, yi = C.tile(C.op.axis[1], C.op.axis[2], bn, bm)

    CC.compute_at(C, yo)

    bc, xc, yc = CC.op.axis

    k, = CC.op.reduce_axis
    ko, ki = CC.split(k, factor=rn)

    CC.reorder(ko, xc, ki, yc)
    CC.unroll(ki)
    CC.vectorize(yc)

    b = C.op.axis[0]
    xb = C.fuse(b, xo)
    C.parallel(xb)

    return s, bufs


def deal_gemm(ops, bufs):
    print("For gemm")
    bs, ns, ks = bufs[0].shape
    bs, ks, ms = bufs[1].shape
    
    best_tile_time = -1
    best_tile_size = []

    tensors = [bufs[0], bufs[1]]
    output_tensor = bufs[2]

    ctx = tvm.cpu(0)
    input_tvm = []

    for tensor in tensors:
        data = np.random.random(
            [int(j) for j in tensor.shape]).astype(np.float32) * 100
        tvm_data = tvm.nd.array(data, ctx)
        input_tvm.append(tvm_data)

    output_holder = tvm.nd.array(
        np.zeros([int(j) for j in output_tensor.shape],
                 dtype=output_tensor.dtype), ctx
    )

    input_tvm = input_tvm + [output_holder]

    for i in range(2,8):
        for j in range(2,8):
            for k in range(0,4):
                if (1<<i)>int(ns) :
                    continue
                if (1<<j)>int(ks) :
                    continue
                if (1<<k)>int(ms) :
                    continue

                new_s, new_bufs = schedule_gemm_with(ops, bufs, 1<<i, 1<<j, 1<<k)

                func = tvm.build(new_s, new_bufs)
                
                evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
                tvm_time = evaluator(*input_tvm).mean * 1e3
                
                print(str(1<<i)+' '+str(1<<j)+' '+str(k))
                print(tvm_time)
                
                if (best_tile_time == -1) | (tvm_time < best_tile_time):
                    best_tile_time = tvm_time
                    best_tile_size = [1<<i, 1<<j, 1<<k]

    i,j,k = best_tile_size
    s, bufs = schedule_gemm_with(ops, bufs, i, j, k)
    return s, bufs
      
def deal_conv(ops, bufs):
      print("For conv")
      s = tvm.create_schedule(ops)
      
      length = len(s.stages)
      #s = tvm.create_schedule(ops)
      #print(s.stages)
      if(length == 6):
        A, B, C, D, E, F = s.stages
      else:
        A, B, C, D = s.stages
        
      #print(B.op.axis)
      #print(D.op.axis) 
      
      bc = B.fuse(B.op.axis[0], B.op.axis[1])
      B.parallel(bc)
      
      bc = D.fuse(D.op.axis[0], D.op.axis[1])
      D.parallel(bc)
      
      if(length == 6):
        #print(F.op.axis)
        bc = F.fuse(F.op.axis[0], F.op.axis[1])
        F.parallel(bc)

      return s, bufs  
      
def new_solve(axisList):
    new_list = axisList[:]
    size = len(axisList)
    while True:
        x1 = random.randint(0,size-1)
        x2 = random.randint(0,size-1)
        if x1 != x2:
            break
    new_list[x1] = axisList[x2]
    new_list[x2] = axisList[x1]
    return new_list

def get_time(s,bufs,input_tvm,ctx,number=3):
    func = tvm.build(s,bufs)
    evaluator = func.time_evaluator(func.entry_name,ctx,number=number)
    tvm_time = 0
    try:
        tvm_time = evaluator(*input_tvm).mean * 1e3
    except Exception as e:
        print(e)
    return int(tvm_time)

def judge(dE,t):
    if (dE < 0):
        return True
    else:
        d = math.exp(-(dE/t))
        if (d > random.random()):
            return True
        else:
            return False
           
# using SA algorithm to schedule gemm reorder        
def schedule_gemm_reorder(s,bufs):
    # create environment for test
    l = len(bufs) 
    tensors = []
    for i in range(l-1):
        tensors.append(bufs[i])

    output_tensor = bufs[l-1]
    ctx = tvm.cpu(0)
    input_tvm = []

    for tensor in tensors:
        data = np.random.random(
            [int(j) for j in tensor.shape]).astype(np.float32) * 100
        tvm_data = tvm.nd.array(data, ctx)
        input_tvm.append(tvm_data)

    output_holder = tvm.nd.array(
        np.zeros([int(j) for j in output_tensor.shape],
                 dtype=output_tensor.dtype), ctx
    )

    input_tvm = input_tvm + [output_holder]

    # split axis with large size
    sA,sB,sC = s.stages
    b,x,y = sC.op.axis
    k, = sC.op.reduce_axis
    xo,yo,xi,yi = sC.tile(x,y,16,16)
    ko,ki = sC.split(k,factor = 4)
    sC.reorder(b,xo,yo,xi,yi,ko,ki)
    old_list = [b,xo,yo,xi,yi,ko,ki]
    start = time.time()
    # using SA algorithm to find best order
    tmp = 1000
    tmp_min = 0.001
    alpha = 0.98
    counter = 0
    counter2 = 0
    old_time = get_time(s,bufs,input_tvm,ctx)
 
    while(tmp > tmp_min):
        new_list = new_solve(old_list)
        sC.reorder(*new_list)
        new_time = get_time(s,bufs,input_tvm,ctx)
        dE = (new_time - old_time)
        j = judge(dE,tmp)
        if j:
            old_time = new_time
            old_list = new_list[:]
        else:
            sC.reorder(*old_list)

        if (dE < 0):
            tmp = tmp * alpha
            counter2 = counter2 + 1
            if (counter2 % 10 == 0):
                print(counter2,": time = ",new_time," temp = ",tmp)

        else:
            counter = counter + 1
        
        if counter > 10000:
            print("tried times exceed boundary, breaking")
            break

    end = time.time()
    print("Tot times tried:",counter + counter2)
    print("Time used for SA:",end-start)
    print("order choosed:",old_list)        
    return s, bufs

def schedule_conv2d_reorder(s,bufs):
    # create environment for test
    l = len(bufs) 
    tensors = []
    for i in range(l-1):
        tensors.append(bufs[i])
    output_tensor = bufs[l-1]
    ctx = tvm.cpu(0)
    input_tvm = []

    for tensor in tensors:
        data = np.random.random(
            [int(j) for j in tensor.shape]).astype(np.float32) * 100
        tvm_data = tvm.nd.array(data, ctx)
        input_tvm.append(tvm_data)

    output_holder = tvm.nd.array(
        np.zeros([int(j) for j in output_tensor.shape],
                 dtype=output_tensor.dtype), ctx
    )

    input_tvm = input_tvm + [output_holder]

    # split axis with large size
    print("scheduling conv2d_nchw")
    length = len(s.stages)
    if (length == 6):
        sinput,spadded,sweight,soutput,sbias,soutputbias = s.stages
    else:
        sinput,spadded,sweight,soutput = s.stages
  
    b,o,h,w = soutput.op.axis
    rc,rw,rh = soutput.op.reduce_axis
    bo,bi  = soutput.split(b,factor = 4)
    ho,wo,hi,wi = soutput.tile(h,w,4,4)
    oo,oi = soutput.split(o,factor = 16)
    rco,rci = soutput.split(rc,factor = 16)
    soutput.reorder(bo,bi,rco,rci,oo,oi,ho,hi,rh,wo,wi,rw)
    old_list = [bo,bi,rco,rci,oo,oi,ho,hi,rh,wo,wi,rw]
    start = time.time()
    # using SA algorithm to find best order
    tmp = 1000
    tmp_min = 0.1
    alpha = 0.98
    counter = 0
    counter2 = 0
    old_time = get_time(s,bufs,input_tvm,ctx)
    ans_time = old_time
    ans_list = old_list
 
    while(tmp > tmp_min):
        new_list = new_solve(old_list)
        soutput.reorder(*new_list)
        new_time = get_time(s,bufs,input_tvm,ctx)
        if (new_time < ans_time):
            ans_time = new_time
            ans_list = new_list

        dE = (new_time - old_time)
        j = judge(dE,tmp)
        if j:
            old_time = new_time
            old_list = new_list[:]
        else:
            soutput.reorder(*old_list)

        if (dE < 0):
            tmp = tmp * alpha
            counter2 = counter2 + 1
            if (counter2 % 10 == 0):
                print(counter2,": time = ",new_time," temp = ",tmp," counter = ",counter)

        else:
            counter = counter + 1
        
        if counter > 25000:
            print("tried times exceed boundary, breaking")
            break

    end = time.time()
    print("Tot times tried:",counter + counter2)
    print("Time used for SA:",end-start)
    print("order choosed:",ans_list) 
    soutput.reorder(*ans_list)       
    return s, bufs

def auto_schedule(func, args):
    """Automatic scheduler
    
    Args:
    -----------------
    func: function object
        similar to batch_gemm function mentioned above
    args: tuple
        inputs to func
    -----------------
    Returns:
    s: tvm.schedule.Schedule
    bufs: list of tvm.tensor.Tensor
    """
    ops, bufs = func(*args)
    
    print(len(bufs[1].shape))
    #print(type(ops))
    #for i in ops:
      #print(type(i))
      #print(i.op)
    #################################################
    # do some thing with `ops`, `bufs` and `args`
    # to analyze which schedule is appropriate
    
    if (func.__name__ == 'batch_gemm'):
      #print("  GEMM   !!!!!!!!")
      s, bufs = deal_gemm(ops, bufs)
    else:
      s, bufs = deal_conv(ops, bufs)
    
    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs 