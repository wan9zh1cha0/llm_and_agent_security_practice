# torch Tensors
* mytensor.shape[-1] represents the last dimension, which is the last value of the array returned by mytensor.shape()
* tril returns the lower triangular part of the matrix (excluding the diagonal), triu returns the upper triangular part (excluding the diagonal)
* torch.stack(a,dim=0) stacks [tensor1, tensor2] side by side; torch.cat(a,dim=1) concatenates them into one
```
>>> a
[tensor([[1, 2, 3],
        [1, 2, 5]]), tensor([[2, 3, 4],
        [3, 4, 5]])]
>>> a1+a2
tensor([[ 3,  5,  7],
        [ 4,  6, 10]])
>>> torch.stack(a,dim=0)
tensor([[[1, 2, 3],
         [1, 2, 5]],

        [[2, 3, 4],
         [3, 4, 5]]])
>>> torch.cat(a,dim=0)
tensor([[1, 2, 3],
        [1, 2, 5],
        [2, 3, 4],
        [3, 4, 5]])
>>> torch.cat(a,dim=1)
tensor([[1, 2, 3, 2, 3, 4],
        [1, 2, 5, 3, 4, 5]])
>>> torch.stack(a, dim=0).shape
torch.Size([2, 2, 3])
```



# torch.nn Neural (Network Related Modules)
* nn.dropout module produces different masks when called consecutively; if you want to fix it, you must reset the random number generator before each use

# torch.utils.data (Dataset)

* torch.utils.data.Dataset 
* torch.utils.data.DataLoader wraps the Dataset iterator

# model
* model.train() switches the model to training mode (training mode); model.train() must be used during the training phase
* model.eval() switches to evaluation mode (evaluation/inference mode); model.eval() must be used during validation/testing/inference phases