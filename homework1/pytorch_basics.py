import torch


class PyTorchBasics:
    """
    Implement the following python code with PyTorch.
    Use PyTorch functions to make your solution efficient and differentiable.

    General Rules:
    - No loops, no function calls (except for torch functions), no if statements
    - No numpy
    - PyTorch and tensor operations only
    - No assignments to results x[1] = 5; return x
    - A solution requires less than 10 PyTorch commands

    The grader will convert your solution to torchscript and make sure it does not
    use any unsupported operations (loops etc).
    """

    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        """
        Return every 3rd element of the input tensor.

        x is a 1D tensor

        --------
        y = []
        for i, v in enumerate(x):
            if i % 3 == 0:
                y.append(v)
        return torch.stack(y, dim=0)
        --------

        Solution length: 13 characters
        """
        return x[0::3]

    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
        """
        Return the maximum value of each row of the final dimension of the input tensor

        x is a 3D tensor

        --------
        n, m, _ = x.shape
        y = torch.zeros(n, m)
        for i in range(n):
            for j in range(m):
                maxval = float("-inf")
                for v in x[i, j]:
                    if v > maxval:
                        maxval = v
                y[i, j] = maxval
        return y
        --------

        Solution length: 26 characters

        --------
        With variables:
        elements, indices = torch.max(x, dim=2)
        return elements
        """
        return torch.max(x, dim=2).values

    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        """
        Return the unique elements of the input tensor in sorted order

        x can have any dimension

        --------
        y = []
        for i in x.flatten():
            if i not in y:
                y.append(i)
        return torch.as_tensor(sorted(y))
        --------

        Solution length: 22 characters

        --------
        With variables:
        flat = torch.flatten(x)
        unique = torch.unique(flat)
        sorted, indices = torch.sort(unique)
        return sorted
        """
        return torch.sort(torch.unique(torch.flatten(x))).values
    
    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the number of elements in y that are greater than the mean of x

        x and y can have any shape

        --------
        a = 0
        b = 0
        for i in x.flatten():
            a += i
            b += 1
        mean = a / b
        c = 0
        for i in y.flatten():
            if i > mean:
                c += 1
        return torch.as_tensor(c)
        --------

        Solution length: 27 characters

        --------
        With variables:
        xMean = torch.mean(x)
        yFlat = torch.flatten(y)
        yGT = torch.gt(yFlat, xMean)
        yToFloat = yGT.to(torch.float32)
        total = torch.count_nonzero(yToFloat)
        return total
        """
        return torch.count_nonzero(torch.gt(torch.flatten(y), torch.mean(x)).to(torch.float32))

    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        """
        Return the transpose of the input tensor

        x is a 2D tensor

        --------
        y = torch.zeros(x.shape[1], x.shape[0])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y[j, i] = x[i, j]
        return y
        --------

        Solution length: 11 characters
        """
        return x.mT

    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        """
        Return the diagonal elements (top left to bottom right) of the input tensor

        x is a 2D tensor

        --------
        y = []
        for i in range(min(x.shape[0], x.shape[1])):
            y.append(x[i, i])
        return torch.as_tensor(y)
        --------

        Solution length: 19 characters
        """
        return torch.diagonal(x)

    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        """
        Return the diagonal elements (top right to bottom left) of the input tensor

        x is a 2D tensor

        --------
        y = []
        for i in range(min(x.shape[0], x.shape[1])):
            y.append(x[i, x.shape[1] - i - 1])
        return torch.as_tensor(y)
        --------

        Solution length: 27 characters

        --------
        With variables:
        flip = torch.flip(x, dims=[1])
        diagonal = torch.diagonal(flip)
        return diagonal
        """
        return torch.diagonal(torch.flip(x, dims=[1]))

    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        """
        Return the cumulative sum of the input tensor

        x is a 1D tensor

        --------
        if len(x) == 0:
            return torch.as_tensor(x)
        y = [x[0]]
        for i in range(1, len(x)):
            y.append(y[i - 1] + x[i])
        return torch.as_tensor(y)
        --------

        Solution length: 22 characters
        """
        return torch.cumsum(x, dim=0)
    
    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of all elements in the rectangle upto (i, j)th element

        x is a 2D tensor

        --------
        y = torch.zeros_like(x)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                y[i, j] = x[i, j]
                if i > 0:
                    y[i, j] += y[i - 1, j]
                if j > 0:
                    y[i, j] += y[i, j - 1]
                if i > 0 and j > 0:
                    y[i, j] -= y[i - 1, j - 1]
        return y
        --------

        Solution length: 36 characters

        --------
        With variables:
        rowSum = torch.cumsum(x, dim=1)
        colSum = torch.cumsum(rowSum, dim=0)
        return colSum
        """
        return torch.cumsum(torch.cumsum(x, dim=1), dim=0)

    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Return the input tensor with all elements less than c set to 0

        x is a 2D tensor
        c is a scalar tensor (dimension 0)

        --------
        y = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] < c:
                    y[i, j] = 0.0
                else:
                    y[i, j] = x[i, j]
        return y
        --------

        Solution length: 49 characters
        """
        return torch.where(x < c, torch.tensor(0), x)
    
    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Return the row and column indices of the elements less than c

        x is a 2D tensor
        c is a scalar tensor (dimension 0)

        The output is a 2 x n tensor where n is the number of elements less than c

        --------
        row, col = [], []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] < c:
                    row.append(i)
                    col.append(j)
        return torch.as_tensor([row, col])
        --------

        Solution length: 30 characters

        --------
        With variables:
        modified = torch.where(x < c, torch.tensor(0), x)
        indices = (modified == 0).nonzero()
        return indices.mT
        """
        return ((torch.where(x < c, torch.tensor(0), x)) == 0).nonzero().mT
        
    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        """
        Return the elements of x where m is True

        x and m are 2D tensors

        --------
        y = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if m[i, j]:
                    y.append(x[i, j])
        return torch.as_tensor(y)
        --------

        Solution length: 11 characters
        """
        return x[m]
    
    @staticmethod
    def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the difference between consecutive elements of the sequence [x, y]

        x and y are 1D tensors

        --------
        xy = []
        for xi in x:
            xy.append(xi)
        for yi in y:
            xy.append(yi)

        z = []
        for xy1, xy2 in zip(xy[1:], xy[:-1]):
            z.append(xy1 - xy2)
        return torch.as_tensor(z)
        --------

        Solution length: 36 characters

        --------
        With variables:
        xy = torch.cat((x, y))
        xy1 = xy[1:]
        xy2 = xy[:-1]
        diff = xy1 - xy2
        return diff
        """
        return (torch.cat((x, y))[1:] - torch.cat((x, y))[:-1])
    

    @staticmethod
    def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Find the number of elements in x that are equal (abs(x_i-y_j) < 1e-3) to at least one element in y

        x and y are 1D tensors

        --------
        count = 0
        for xi in x:
            for yi in y:
                if abs(xi - yi) < 1e-3:
                    count += 1
                    break
        return torch.as_tensor(count)
        --------

        Solution length: 64 characters

        --------
        With variables:
        xBroad = x[:, None]
        yBroad = y[None, :]
        calc = torch.abs(xBroad - yBroad)
        check = calc < torch.tensor(1e-3)
        count = check.any(dim=1).sum()
        return count
        """
        return (torch.abs(x[:, None] - y[None, :]) < torch.tensor(1e-3)).any(dim=1).sum()