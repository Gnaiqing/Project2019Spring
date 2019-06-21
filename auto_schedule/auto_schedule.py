import tvm
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
    
    if(len(bufs[1].shape) == 3):
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