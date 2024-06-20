import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    import optims
except ImportError as IE:
    print("Cannot import Implicit Competitive Regularization package")
    print(IE)


__OPTIMIZERS__ = ['ADAM', 'SGD', 'RMSprop', 'ADAgrad', 'ADAmax', 'ICR', 'ACGD']


class BaseOptimizer:
    r"""
    Loss criterion class. Must define 4 members:
    sizeDecisionLayer : size of the decision layer of the discrimator

    getCriterion : how the loss is actually computed

    !! The activation function of the discriminator is computed within the
    loss !!
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, loss):
        pass

    def init_optimizer(self, lr, G, D):
        r"""
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss

        Args:

            - input (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        pass

    def zero_grad(self):
        pass


class ADAM(BaseOptimizer):
    r"""
    Mean Square error loss.
    """

    def __init__(self, device, betas=[0, 0.99]):
        self.betas = betas

        BaseOptimizer.__init__(self, device)

    def init_optimizer(self, G, D, lr):
        if type(lr) is list:
            self.lrG = lr[0]
            self.lrG = lr[1]
        else:
            self.lrG = self.lrD = lr

        self.optimizerG = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()),
                          betas=self.betas, lr=self.lrG)
        self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),
                          betas=self.betas, lr=self.lrD)

    def step(self, loss, is_g=False):
        loss.backward()
        if is_g:
            self.optimizerG.step()
        else:
            self.optimizerD.step()

    def zero_grad(self):
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()


class SGD(BaseOptimizer):
    r"""
    Mean Square error loss.
    """

    def __init__(self, device, momentum=0.9):
        self.momentum = momentum

        BaseOptimizer.__init__(self, device)

    def init_optimizer(self, G, D, lr):
        if type(lr) is list:
            self.lrG = lr[0]
            self.lrG = lr[1]
        else:
            self.lrG = self.lrD = lr

        self.optimizerG = optim.SGD(filter(lambda p: p.requires_grad, G.parameters()),
                          momentum=self.momentum, lr=self.lrG)
        self.optimizerD = optim.SGD(filter(lambda p: p.requires_grad, D.parameters()),
                          momentum=self.momentum, lr=self.lrD)

    def step(self, loss, is_g=False):
        loss.backward()
        if is_g:
            self.optimizerG.step()
        else:
            self.optimizerD.step()

    def zero_grad(self):
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()


class RMSprop(BaseOptimizer):
    r"""
    Mean Square error loss.
    """

    def __init__(self, device, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

        BaseOptimizer.__init__(self, device)

    def init_optimizer(self, G, D, lr):
        if type(lr) is list:
            self.lrG = lr[0]
            self.lrG = lr[1]
        else:
            self.lrG = self.lrD = lr

        self.optimizerG = optim.RMSprop(filter(lambda p: p.requires_grad, G.parameters()),
                                        alpha=self.alpha, 
                                        lr=self.lrG,
                                        momentum=self.momentum,
                                        eps=self.eps,
                                        centered=self.centered,
                                        weight_decay=self.weight_decay)
        self.optimizerD = optim.RMSprop(filter(lambda p: p.requires_grad, D.parameters()),
                                        alpha=self.alpha, 
                                        lr=self.lrG,
                                        momentum=self.momentum,
                                        eps=self.eps,
                                        centered=self.centered,
                                        weight_decay=self.weight_decay)

    def step(self, loss, is_g=False):
        loss.backward()
        if is_g:
            self.optimizerG.step()
        else:
            self.optimizerD.step()

    def zero_grad(self):
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()


class ICR(BaseOptimizer):
    r"""
    Implicit Competitive Regularization optimization.
    """

    def __init__(self, device):
        BaseOptimizer.__init__(self, device)

    def init_optimizer(self, G, D, lr):
        if type(lr) is list:
            self.lrG = lr[0]
            self.lrG = lr[1]
        else:
            self.lrG = self.lrD = lr

        self.optimizer = optims.ICR(max_params=filter(lambda p: p.requires_grad, G.parameters()),
                                  min_params=filter(lambda p: p.requires_grad, D.parameters()),
                                  lr=self.lrG,
                                  device=self.device)
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, loss, is_g=False):
        if not is_g:
            self.optimizer.step(loss)


class ACGD(BaseOptimizer):
    r"""
    Adaptive Competitive Gradient Descent optimization.
    """

    def __init__(self, device):
        BaseOptimizer.__init__(self, device)

    def init_optimizer(self, G, D, lr):
        if type(lr) is list:
            self.lrG = lr[0]
            self.lrG = lr[1]
        else:
            self.lrG = self.lrD = lr

        self.optimizer = optims.ACGD(max_params=filter(lambda p: p.requires_grad, G.parameters()),
                              min_params=filter(lambda p: p.requires_grad, D.parameters()),
                              lr_max=self.lrG,
                              lr_min=self.lrD,
                              device=self.device)
        

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, loss, is_g=False):
        if not is_g:
            self.optimizer.step(loss)
