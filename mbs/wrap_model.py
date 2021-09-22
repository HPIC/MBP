from typing import Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

class MBSModel:
    def __init__(
        self, module : Union[Module, Sequential], mbs
    ) -> None:
        self._comm_mbs = mbs


class MBSBatchNorm(_BatchNorm):
    r'''
        MBS::BatchNorm
        
        the original PyTorch::BatchNorm formulation:
            (ref: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
            >>> x_new = ( 1 - momentum ) * x_bar + momentum * x_t
            (x_bar: estimated statistic, x_t: new observed value )
        we need to accumulate new micro-batch observed value at each micro-batch iteration.

        how to normalize? (if mini-batch size is 64, micro-batch size is 16)
            u-iter0 ~ u-iter3 :
                calculate and accumulate Sigma(u-input) & Sigma(u-input**2) until final u-iter.
                and also accumulate u-input size.
            u-iter3:
                update new batch normalization using accumulated values(Sigma(u-input) & Sigma(u-input**2)).
                new mean <- Sigma(u-input) / accumulated u-input size.
                new var <- Sigma(u-input**2) / accumulated u-input size - new mean.
                update method is like that:
                    >>> x_new = ( 1 - momentum ) * x_bar + momentum * x_t

        Args:
            copy PyTorch::BatchNorm parameter written by user.
    '''
    def __init__(
        self, num_features, eps=0.00001, momentum=0.1, affine=True,
        mbs = None,
    ) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=True
        )
        self.register_buffer('accum_sum', torch.zeros_like(self.running_mean))
        self.register_buffer('accum_sum_squares', torch.zeros_like(self.running_var))

        self._comm_mbs = mbs
        self.accum_size = 0

    def _check_input_dim(self, input: Tensor):
        if input.dim() <= 2:
            raise ValueError('[MBS error] expected at least 3D input (got %dD input)' % input.dim())

    @torch.no_grad()
    def _accumulate(self, input: Tensor):
        '''
            calculate and accumulate Sigma(u-input) & Sigma(u-input**2) until final u-iter.
            and also accumulate u-input size.

            Args:
                input : torch.Tensor
        '''
        dim = [0]
        dim.extend( range( 2, input.dim() ) )

        self.accum_sum += input.sum(dim)
        self.accum_sum_squares += (input**2).sum(dim)

        size = input.size().numel() // input.size(1)
        self.accum_size += size

    def _normalize(self):
        '''
            update new batch normalization using accumulated values(Sigma(u-input) & Sigma(u-input**2)).
        '''
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum

        mean = self.accum_sum / self.accum_size # E[X] = sum(X) / size
        var = self.accum_sum_squares / self.accum_size - mean**2 # Var[X] = E[X^2] - E[X]^2, E[X^2] = sum(X^2) / size.

        self.running_mean *= ( 1 - exponential_average_factor )
        self.running_mean += ( mean * exponential_average_factor )

        self.running_var *= ( 1 - exponential_average_factor )
        self.running_var += ( var * exponential_average_factor )

        self.accum_sum.zero_()
        self.accum_sum_squares.zero_()
        self.accum_size = 0

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        # Test mode
        if not self.training:
            return F.batch_norm(
                input,
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )

        # Training mode
        self._accumulate(input)

        if self._comm_mbs._update_timing:
            self._check_input_dim(input)
            self._normalize()

        return F.batch_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=self.weight,
            bias=self.bias,
            training=True,
            momentum=0.0, # momentum 0 means that do not accumulate.
            eps=self.eps,
        )

    @classmethod
    def wrap_batch_norm(cls, module: Module, mbs):
        '''
            wrap PyTorch::BatchNorm to MBS::BatchNorm.
            it means only replace PyTorch::BatchNorm to MBS::BatchNorm.

            Args:
                module : Pytorch::Module
                mbs : Micro Batch Streaming object

            Returns:
                module_output : Pytorch::Module
                    but Pytorch::BatchNorm layers are replaced MBS::BatchNorm.
        '''
        if isinstance(module, MBSBatchNorm):
            return module

        module_output : Module = module

        if isinstance(module, _BatchNorm):
            module_output = MBSBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                mbs
            )

            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer('num_batches_tracked', module.num_batches_tracked)

        for name, mod in module.named_children():
            module_output.add_module(
                name, cls.wrap_batch_norm(mod, mbs)
            )

        return module_output

