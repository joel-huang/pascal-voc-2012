# |  ||
# || |_
import torch
from torch.nn import LogSigmoid
from torch.nn.functional import _Reduction
from torch.nn.modules.loss import _WeightedLoss
from torch._jit_internal import weak_module, weak_script_method

nb_scale = 1e-2
logsigmoid = LogSigmoid()

def multilabel_nb_loss(nb_mat, input, target, weight=None, size_average=None,
                                reduce=None, reduction='mean', scaling_c=nb_scale):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    # multilabel_nb_loss(input, target, weight=None, size_average=None) -> Tensor
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    # An example of target labels: [person, horse, car] -> one-hot encoding
    # For each class present in the true label, look up P(all labels|class).
    # This corresponds to a full column in nb_mat (The printed, transposed
    # version has them as rows instead).

    # The loss for a particular class x decreases when there are occurrences
    # of other classes c. Therefore a high value of P(x|c) will contribute
    # to decreased loss for class x.

    
    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    # For each sample, find the class indices (where target is 1).
    # If there is more than one class, update the loss via this equation:
    # loss = loss - nb_mat[c,x], where c is found in company of x.
    for row in range(target.shape[0]):
        class_indices = target[row].nonzero()
        if len(class_indices) > 1:
            for x in class_indices:
                for c in class_indices:
                    if not torch.equal(x, c):
                        loss[row, x] -= nb_mat[c,x]*scaling_c

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    torch.set_printoptions(profile="default")
    return ret

class MultiLabelNBLoss(_WeightedLoss):
    """
    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """
    __constants__ = ['weight', 'reduction']

    def __init__(self, mat, weight=None, size_average=None, reduce=None, reduction='mean', scaling_c=nb_scale):
        super(MultiLabelNBLoss, self).__init__(weight, size_average, reduce, reduction)
        self.nb_mat = mat
        self.scaling_c = scaling_c

    @weak_script_method
    def forward(self, input, target):
        return multilabel_nb_loss(self.nb_mat, input, target, weight=self.weight, reduction=self.reduction, scaling_c=self.scaling_c)