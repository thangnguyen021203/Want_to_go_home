import torch.nn.functional as F
from federated.fedavg import aggregate
import torch

class Server:
    def __init__(self, model, workers, test_dataloader=None):
        self.global_model = model
        self.workers = workers
        self.test_dataloader = test_dataloader

    def update_model(self, client_updates):
        """
        Gọi hàm aggregate để tổng hợp mô hình toàn cục.
        """
        aggregate(self.global_model, client_updates)

    def evaluate(self):
        """
        Đánh giá mô hình toàn cục.
        """
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Giả sử dữ liệu validation có sẵn trên một worker (hoặc tập trung)
        for data, labels in self.test_dataloader:
            output = self.global_model(data)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        total_loss /= total
        accuracy = 100.0 * correct / total
        return loss, accuracy