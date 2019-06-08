import tvm
from tvm import autotvm
import numpy as np
# using cfg to create schedule for gemm        
def schedule_gemm(ops,bufs,cfg):
    s = tvm.create_schedule(ops)
    block_cfg = cfg["block"]
    param_cfg = cfg["param"]
    if (block_cfg[0]):        
        A,B,C = bufs
        bn = param_cfg[0]
        bm = param_cfg[1] 
        fc = param_cfg[2]
        CC = s.cache_write(C,'global')
        xo, yo, xi, yi = s[C].tile(C.op.axis[1],C.op.axis[2],bn,bm)
        # allocate write cache
        s[CC].compute_at(s[C],yo)
        bc,xc,yc = s[CC].op.axis
        k, = s[CC].op.reduce_axis        
        ko,ki = s[CC].split(k,factor = fc)
        s[CC].reorder(ko,xc,ki,yc)
        s[CC].unroll(ki)
        s[CC].vectorize(yc)
        
        b = s[C].op.axis[0]
        xb = s[C].fuse(b,xo)
        s[C].parallel(xb)

    return s,bufs
        
def deal_gemm(ops, bufs):
    print("For gemm")
    bs, ns, ks = bufs[0].shape # A
    bs, ks, ms = bufs[1].shape # B
    
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

    cfg = {
            "block":{0:1},
            "param":{}
        }
    space = {}
    space[0] = [16,32,64,128,256] # tile_x
    space[1] = [16,32,64,128,256] # tile_y
    space[2] = [1,2,4,8,16] #split_factor
    for i in space[0]:
        for j in space[1]:
            for k in space[2]:
                if (i > int(ns)) or (j > int(ks)) or (k > int(ms)) :
                    continue
                cfg["param"][0] = i
                cfg["param"][1] = j
                cfg["param"][2] = k
                new_s, new_bufs = schedule_gemm(ops, bufs, cfg)

                func = tvm.build(new_s, new_bufs)
                
                evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
                tvm_time = evaluator(*input_tvm).mean * 1e3
                
                print('bn='+str(i)+' bm='+str(j)+' fc='+str(k))
                print(tvm_time)
                
                if (best_tile_time == -1) | (tvm_time < best_tile_time):
                    best_tile_time = tvm_time
                    best_tile_size = [i,j,k]
                    ans_s = new_s
                    ans_bufs = new_bufs
               
    print("best time = ",best_tile_time)
    print("best size = ",best_tile_size)
    return ans_s, ans_bufs
        
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
    #################################################
    # do some thing with `ops`, `bufs` and `args`
    # to analyze which schedule is appropriate
    
    s = tvm.create_schedule(ops)
    # size = len(bufs)
    # for tensor in bufs:
    #     print(tensor.op.input_tensors)

    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    if (func.__name__ == 'batch_gemm'):
        print("scheduling batch_gemm")
        sA,sB,sC = s.stages
        xo, yo, xi, yi = sC.tile(sC.op.axis[1],sC.op.axis[2],32,32)
        k, = sC.op.reduce_axis      
        ko,ki = sC.split(k,factor = 4)
        sC.reorder(xo,yo,xi,ko,ki,yi)
 
    elif (func.__name__ == 'conv2d_nchw'):
        print("scheduling conv2d_nchw")
        length = len(s.stages)
        if (length == 6):
            sinput,spadded,sweight,soutput,sbias,soutputbias = s.stages
        else:
            sinput,spadded,sweight,soutput = s.stages
  
        b,o,h,w = soutput.op.axis # [batch_size,out_channel,out_h,out_w]
        rc,rw,rh = soutput.op.reduce_axis
        soutput.reorder(b,o,rc,h,rh,w,rw)
               
    print(tvm.lower(s,bufs,simple_mode=True))
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs
