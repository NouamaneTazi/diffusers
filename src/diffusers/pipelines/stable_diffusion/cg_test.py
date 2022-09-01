# dlprof --nsys_opts="-t cuda,nvtx" --mode=pytorch --output_path=./dlprof_logs/test_cg_1 python H3C/benchmarks/bert/implementations/pytorch/function.py

# python function.py [--graph-after-ddp] [--graph-before-ddp]
# python -m torch.distributed.launch --nproc_per_node=2 function.py [--graph-after-ddp] [--graph-before-ddp]

import torch
import types
from itertools import chain
import argparse
import os

# questions:
# is a custom autograd function or graphing around a backward call better?
# how to allow double backward?
# lazily capture as part of live backward, or not?
# capture all the way down to AccumulateGrad functions, or not?
# If yes, need to deal with params used in graphs and non-graphed regions,
# and DDP bucket-slot-ready flags.  To help, user could supply a list of params
# known to be exclusive to the graphed region.

# Current limitation:  Assumes all args are Tensors.
# Arg tensors may or may not require grad.
# Any temporaries created in func_or_module must not be used
# outside func_or_module unless they are among func_or_module's
# explicit return values.
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime

def graph(func_or_module,
          sample_args,
          sample_args_eval=None,
          graph_stream=None,
          warmup_iters=3,
          warmup_only=False):

    assert isinstance(sample_args, tuple)

    # To run a module's forward method as a torch.autograd.Function,
    # and ensure gradients of all used tensors are returned by the Function's backward
    # so the autograd engine takes care of final accumulation (which includes DDP hooks)
    # we need to "functionalize" module.forward:
    # createa a wrapper function where module attributes
    # and user args all enter through the arglist.
    was_module = isinstance(func_or_module, torch.nn.Module)
    if was_module:
        if isinstance(func_or_module, torch.nn.parallel.DistributedDataParallel):
            func_or_module = func_or_module.module
        module_params = tuple(func_or_module.parameters())
        functional_args = sample_args + module_params

    stream = torch.cuda.Stream() if graph_stream is None else graph_stream
    ambient_stream = torch.cuda.current_stream()
    stream.wait_stream(ambient_stream)

    # Most of the spaghetti here comes from handling args that may not require grad.

    with torch.cuda.stream(stream):
        # Capture eval
        capture_eval = (sample_args_eval is not None)
        if capture_eval:
            assert isinstance(sample_args_eval, tuple)
            assert len(sample_args_eval) == len(sample_args)

            with torch.no_grad():
                func_or_module.eval()

                # warmup iters before capture
                for _ in range(warmup_iters):
                    eval_outputs  = func_or_module(*sample_args_eval)
                    eval_outputs_was_tensor = isinstance(eval_outputs, torch.Tensor)
                    eval_outputs = (eval_outputs,) if eval_outputs_was_tensor else eval_outputs

                if warmup_iters > 0:
                    del eval_outputs

                print("Eval-Graphing\n", flush=True)

                eval_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(eval_graph):
                    eval_outputs  = func_or_module(*sample_args_eval)

                eval_outputs_was_tensor = isinstance(eval_outputs, torch.Tensor)
                eval_outputs = (eval_outputs,) if eval_outputs_was_tensor else eval_outputs


    ambient_stream.wait_stream(stream)

    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            with torch.no_grad():
                if capture_eval:
                    for i, arg in zip(sample_args_eval, inputs[0:len(sample_args)]):
                        # assert i.shape == arg.shape, "eval capture shape doesn't match run input shape"
                        if i.data_ptr() != arg.data_ptr():
                            i.copy_(arg)
                    eval_graph.replay()
                    return eval_outputs
                else:  # execute eval eagerly
                    outputs = func_or_module.forward_eager(*inputs[0:len(sample_args)])
                    if not isinstance(outputs, tuple):
                        outputs = (outputs,)
                    return outputs

    if was_module:
        def functionalized(self, *user_args):
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if eval_outputs_was_tensor else out
        func_or_module.forward_eager = func_or_module.forward
        func_or_module.forward = types.MethodType(functionalized, func_or_module)
        return func_or_module
    else:
        return Graphed.apply


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--graph-before-ddp", default=True, action="store_true")
    parser.add_argument("--graph-after-ddp", default=False, action="store_true")
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank + 1)
    torch.cuda.manual_seed(args.local_rank + 1)

    print("{} graph_before_ddp {} graph_after_ddp {}\n".format(args.local_rank,
                                                               args.graph_before_ddp,
                                                               args.graph_after_ddp),
          flush=True)

    N, D_in, H, D_out = 640, 4096, 2048, 1024

    stream = torch.cuda.Stream()

    model_segment1 = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                torch.nn.Dropout(p=0.2),
                                torch.nn.Dropout(p=0.4)).cuda()

    model_segment2 = torch.nn.Sequential(torch.nn.Linear(H, D_out),
                                torch.nn.Dropout(p=0.3),
                                torch.nn.Dropout(p=0.1)).cuda()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(chain(model_segment1.parameters(),
                                      model_segment2.parameters()),
                                lr = 0.1)

    x = torch.randn(N, D_in, device='cuda')
    h = torch.randn(N, H, device='cuda')
    y = torch.randn(N, D_out, device='cuda')

    x_eval = torch.randn(2*N, D_in, device='cuda')
    h_eval = torch.randn(2*N, H, device='cuda')
    y_eval = torch.randn(2*N, D_out, device='cuda')

    pure_eager = not (args.graph_before_ddp or args.graph_after_ddp)

    if args.graph_before_ddp or pure_eager:
        print("Calling graph() before ddp\n")
        model_segment1 = graph(model_segment1,
                               (x.clone(),),
                               (x_eval.clone(),),
                               graph_stream=stream,
                               warmup_only=pure_eager)

        model_segment2 = graph(model_segment2,
                               (h.clone().requires_grad_(),),
                               (h_eval.clone().requires_grad_(),),
                               graph_stream=stream,
                               warmup_only=pure_eager)

    model = torch.nn.Sequential(model_segment1, model_segment2)
    if args.distributed:
        # Small bucket cap to stress DDP
        torch.cuda.nvtx.range_push("DDP")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          bucket_cap_mb=1,
                                                          device_ids=[args.local_rank],
                                                          gradient_as_bucket_view=True)
        torch.cuda.nvtx.range_pop()

    if args.graph_after_ddp:
        if args.distributed:
            print("Calling graph() after ddp\n")
            model.module[0] = graph(model.module[0], (x.clone(),), stream)
        else:
            model[0] = graph(model_segment1, (x.clone(),), stream)

    for e in range(2):
        model.train()
        for i in range(10):
            torch.cuda.nvtx.range_push("{}".format(i))
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()

            # possibly needed if post-backward sync is commented out in pytorch
            # torch.cuda.synchronize()

            torch.cuda.nvtx.range_push("step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

        print("train: {} {} {} {}".format(args.local_rank,
                                        loss.item(),
                                        tuple(p.grad.sum().item() for p in model_segment1.parameters()),
                                        tuple(p.grad.sum().item() for p in model_segment2.parameters())),
            flush=True)

        # do eval end of epoch
        with torch.no_grad():
            model.eval()
            y_pred = model(x_eval)
            loss = loss_fn(y_pred, y_eval)
        print("eval: {} {}".format(args.local_rank,
                                loss.item()),
            flush=True)

if __name__ == "__main__":
    main()
