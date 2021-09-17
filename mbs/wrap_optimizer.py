from torch.optim.optimizer import Optimizer

class MBSOptimizer:
    def __init__(
        self, optimizer : Optimizer, mbs
    ) -> None:
        r'''
            MBSOptimizer, like OOP interface.
            does not inherit torch.optim.Optimizer.

            Warning:
                If you use Scheduler to optimize lr value when your model training,
                You need to call MBSOptimizer class inner variable as below example code.
                Because MBSOptimizer does not inherit torch.optim.Optimizer.
                *** MicroBatchStream only running like OOP interface. ***

                >>> mbs = MicroBatchStream()
                >>> opt = torch.optim.SGD(...)
                >>> opt = mbs.set_optimizer(opt)
                >>> sch = ExponentialLR( opt.optimizer, ... )

            Args:
                optimizer : torch.optim.optimizer.Optimizer
                mbs : Micro Batch Streaming object
                    this is to share data between MBS subclass like MBSLoss, MBSDataloader, MBSOptimizer.
        '''
        self._comm_mbs = mbs
        self.optimizer = optimizer

    def zero_grad(self, _epoch : int = None):
        if self._comm_mbs._zero_grad_timing:
            # print(f'[{_epoch}] set zero gradients.')
            self.optimizer.zero_grad()

    def step(self, _epoch : int = None):
        if self._comm_mbs._update_timing:
            # print(f'[{_epoch}] update parameters.')
            self.optimizer.step()
