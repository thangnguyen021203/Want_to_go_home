import torch

class Client:
    def __init__(self, client_id, model, dataset, config, worker):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.config = config
        self.worker = worker

    def train(self):
        """
        Huấn luyện mô hình trên worker.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(self.config["epochs"]):
            # print(self.dataset.dataset)
            for data, target in self.dataset:
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

        # Trả về trạng thái của mô hình sau khi huấn luyện
        return self.model.state_dict()
