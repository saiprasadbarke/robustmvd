# torch.backends.cudnn.benchmark = True  # type: ignore


class Training:
    def __init__(self, model, dataset, optimizer, scheduler, loss_fct, out_dir):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fct = loss_fct
        self.out_dir = out_dir

    def train(self):
        pass

    def train_loop(self):
        pass
