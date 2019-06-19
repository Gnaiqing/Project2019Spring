import tvm,random,math,time
from tvm import autotvm
import numpy as np

def generate_schedule(ops,stage_id,splitSize,order):
    s = tvm.create_schedule(ops)
    stage = s.stages[stage_id]
    axisList = stage.op.axis[:]
    axisList.extend(stage.op.reduce_axis)
    axisNum = len(axisList)
    tmpList = []
    
    for i in range(axisNum):
        oi,ii = stage.split(axisList[i],factor = splitSize[i])
        tmpList = tmpList + [oi,ii]
    
    finalList = []
    for i in range(2*axisNum):
        finalList.append(tmpList[order[i]])
    
    stage.reorder(*finalList)
    return s
        
def new_solve(axisNum,splitSize,order,tmp):
    r =random.random()
    if (tmp < 1 and r< 0.5): # change split size of an axis
        lenList = [4,8,16,32,64]
        axis = random.randint(0,axisNum-1)
        while True:
            l = lenList[random.randint(0,len(lenList)-1)]
            if (l != splitSize[axis]): 
                break
            
        splitSize[axis] = l
    else:
        while True:
            p1 = random.randint(0,2*axisNum - 1)
            p2 = random.randint(0,2*axisNum - 1)
            if p1 != p2:
                break
            
        tmp = order[p1]
        order[p1] = order[p2]
        order[p2] = tmp
                
    return

def get_time(s,bufs,input_tvm,ctx,number=1):
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
            
# using SA algorithm to schedule op        
def schedule_op(ops,bufs,maxTimes = 10000):
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

    # do schedule for the stage with most reduce_axis
    s = tvm.create_schedule(ops)
    stageList = s.stages
    stageNum = len(stageList)
    reduce_num = -1
    stage_id = -1
    for i in range(stageNum):
        if hasattr(stageList[i].op,'reduce_axis'):
            reduceOpList = stageList[i].op.reduce_axis
            if int(len(reduceOpList)) > reduce_num:
                reduce_num = int(len(reduceOpList))
                stage_id = i
            
    stage = stageList[stage_id]
    print("stage choosed: stage ",stage_id)
    axisList = stage.op.axis[:]
    axisList.extend(stage.op.reduce_axis)
    axisNum = len(axisList)
    splitSize = []
    for i in range(axisNum):
        splitSize.append(4)
        
    order = []
    for i in range(2*axisNum):
        order.append(i)
    
    old_s = generate_schedule(ops,stage_id,splitSize,order)
    old_time = get_time(old_s,bufs,input_tvm,ctx)
    start = time.time()
    # using SA algorithm to find best order
    tmp = 1000
    tmp_min = 0.1
    alpha = 0.98
    counter = 0
    counter2 = 0
 
    while(tmp > tmp_min):
        new_solve(axisNum,splitSize,order,tmp)
        # print(splitSize)
        # print(order)    
        new_s = generate_schedule(ops,stage_id,splitSize,order)
        new_time = get_time(new_s,bufs,input_tvm,ctx)
        dE = (new_time - old_time)
        j = judge(dE,tmp)
        if j:
            old_time = new_time
            old_s = new_s
            old_splitSize = splitSize[:]
            old_order = order[:]

        if (dE < 0):
            tmp = tmp * alpha
            counter2 = counter2 + 1
            if (counter2 % 10 == 0):
                print(counter2,": time = ",new_time," temp = ",tmp," counter = ",counter)

        else:
            counter = counter + 1
        
        if counter > maxTimes:
            print("tried times exceed boundary, breaking")
            break

    end = time.time()
    print("Tot times tried:",counter + counter2)
    print("Time used for SA:",end-start)
    print("split size:", old_splitSize)
    print("order:", old_order)
    return old_s, bufs

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

    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    if (func.__name__ == 'batch_gemm'):
        print("scheduling gemm")
        s,bufs = schedule_op(ops,bufs,10000)
        '''
        sA,sB,sC = s.stages
        b,x,y = sC.op.axis
        k, = sC.op.reduce_axis
        xo,yo,xi,yi = sC.tile(x,y,16,16)
        ko,ki = sC.split(k,factor = 4)
        sC.reorder(xo,b,ko,xi,yo,yi,ki)
        # sC.reorder(b,xo,yo,ko,xi,ki,yi) # used by tutorial
        '''
        
    elif (func.__name__ == 'conv2d_nchw'):
        print("scheduling conv2d_nchw")
        s,bufs = schedule_op(ops,bufs,10000)
        '''
        b,o,h,w = soutput.op.axis
        rc,rw,rh = soutput.op.reduce_axis
        bo,bi  = soutput.split(b,factor = 4)
        ho,wo,hi,wi = soutput.tile(h,w,4,4)
        oo,oi = soutput.split(o,factor = 16)
        rco,rci = soutput.split(rc,factor = 16)
        soutput.reorder(rco,bi,bo,oo,ho,oi,rci,rh,rw,wo,wi,hi)
        '''

    print(tvm.lower(s,bufs,simple_mode=True))
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs
