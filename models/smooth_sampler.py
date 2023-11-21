import torch
import smooth_sampler_cpp


class SmoothSamplerBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, grad_out, align_corners=True, apply_smoothstep=True):
        ctx.align_corners = align_corners
        ctx.apply_smoothstep = apply_smoothstep
        grad_input, grad_grid = smooth_sampler_cpp.backward(grad_out, input, grid, ctx.align_corners, apply_smoothstep, input.requires_grad)
        ctx.save_for_backward(input, grid, grad_out)

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_out_input, grad_out_grid):
        input, grid, grad_out = ctx.saved_tensors

        if grad_out_input is None or (grad_out_input == 0.).all().item():
            grad_input, grad_grid, grad_grad_out = smooth_sampler_cpp.backward_backward_grid_only(grad_out_grid, input, grid, grad_out,
                                                                                                  ctx.align_corners, ctx.apply_smoothstep)
        else:
            grad_input, grad_grid, grad_grad_out = smooth_sampler_cpp.backward_backward(grad_out_input, grad_out_grid, input, grid,
                                                                                        grad_out, ctx.align_corners, ctx.apply_smoothstep)
        return grad_input, grad_grid, grad_grad_out, None, None

class SmoothSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, align_corners=True, apply_smoothstep=True):
        output = smooth_sampler_cpp.forward(input, grid, align_corners, apply_smoothstep)
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.apply_smoothstep = apply_smoothstep
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors

        if (grad_out == 0.).all().item():
            return torch.zeros_like(input), torch.zeros_like(grid), None, None

        d_input, d_grid = SmoothSamplerBackward.apply(input, grid, grad_out.contiguous(), ctx.align_corners, ctx.apply_smoothstep)
        return d_input, d_grid, None, None

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.cuda.manual_seed(3)
    torch.set_printoptions(threshold=10_000)
    input = (torch.rand([2,2,2,3,11], device="cuda:1")).requires_grad_(True)
    grid = (torch.rand([2,2,1,5,3], device="cuda:1") * 2. - 1.).requires_grad_(True)

    out1 = SmoothSampler.apply(input, grid, False, False)
    out2 = torch.nn.functional.grid_sample(input, grid, align_corners=False, padding_mode="border")
    assert torch.allclose(out1, out2)

    grad1_input, grad1_grid = torch.autograd.grad(out1, [input, grid], torch.ones_like(out1))
    grad2_input, grad2_grid = torch.autograd.grad(out2, [input, grid], torch.ones_like(out2))
    assert torch.allclose(grad1_input, grad2_input)
    assert torch.allclose(grad1_grid, grad2_grid)

    input = (torch.rand([2,2,2,3,11], device="cuda").double()).requires_grad_(True)
    grid = (torch.rand([2,2,1,5,3], device="cuda") * 2. - 1.).double().requires_grad_(True)

    torch.autograd.gradcheck(SmoothSampler.apply, [input, grid, False, False], eps=1e-4, atol=1e-3, rtol=1e-2)
    torch.autograd.gradgradcheck(SmoothSampler.apply, [input, grid, False, False], eps=1e-4, atol=1e-3, rtol=1e-2)

    torch.autograd.gradcheck(SmoothSampler.apply, [input, grid, False, True], eps=1e-4, atol=1e-3, rtol=1e-2)
    torch.autograd.gradgradcheck(SmoothSampler.apply, [input, grid, False, True], eps=1e-4, atol=1e-3, rtol=1e-2)

    torch.autograd.gradcheck(SmoothSampler.apply, [input, grid, True, True], eps=1e-4, atol=1e-3, rtol=1e-2)
    torch.autograd.gradgradcheck(SmoothSampler.apply, [input, grid, True, True], eps=1e-4, atol=1e-3, rtol=1e-2)

    torch.autograd.gradcheck(SmoothSampler.apply, [input, grid, True, False], eps=1e-4, atol=1e-3, rtol=1e-2)
    torch.autograd.gradgradcheck(SmoothSampler.apply, [input, grid, True, False], eps=1e-4, atol=1e-3, rtol=1e-2)