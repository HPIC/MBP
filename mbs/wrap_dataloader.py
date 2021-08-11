import math


class wrap_dataset:
    def __init__(self, _dataloader, _micro_batch_size) -> None:
        self.dataloader = _dataloader
        self.micro_batch_size = _micro_batch_size

    def __iter__(self):
        count_update = 0
        count_zero = 0
        zero = False
        update = False
        for _, data in enumerate(self.dataloader):
            A = data["A"]
            B = data["B"]

            num_micro_batch = math.ceil(A.size(0) / self.micro_batch_size)

            As = A.chunk(num_micro_batch)
            Bs = B.chunk(num_micro_batch)

            for a, b in zip(As, Bs):
                count_update += 1
                count_zero += 1

                update = True if count_update % num_micro_batch == 0 else False
                zero = True if count_zero % num_micro_batch == 1 else False

                yield (zero, update, a, b)
