from config import CONFIG
from models.cnn_model import CNNModel
from clients.client import Client
from clients.server import Server
from federated.fedavg import fedavg
from utils.data_utils import get_dataloaders, get_test_dataloader, choose_dataset
from utils.parser import args_parser
import torch
import syft as sy

def setup_virtual_workers(num_workers):
    """
    Thiết lập các Virtual Workers sử dụng VirtualMachine từ PySyft.
    """
    workers = []
    for i in range(num_workers):
        vm = sy.Worker(name=f"worker_{i}")
        worker = vm.get_guest_client() 
        workers.append(worker)
    return workers

def assign_data_to_workers(workers, datasets):
    """
    Gán dataset cho mỗi worker.
    """
    for worker, dataset in zip(workers, datasets):
        # Mỗi worker giữ dataset cục bộ
        worker.dataset = dataset

def main():

    args = args_parser()

    # Load dữ liệu
    train_dataset, test_dataset = choose_dataset(args.dataset)
    train_dataloader = get_dataloaders(train_dataset, 
                                      CONFIG["num_clients"], 
                                      CONFIG["batch_size"], 
                                      args.iid)
    test_dataloader = get_test_dataloader(test_dataset, 
                                          CONFIG["batch_size"])

    # Khởi tạo các worker
    workers = setup_virtual_workers(CONFIG["num_clients"])

    # # Chia dữ liệu cho các client
    # datasets = torch.utils.data.random_split(
    #     train_dataloader.dataset, 
    #     [len(train_dataloader.dataset) // CONFIG["num_clients"]] * CONFIG["num_clients"]
    # )

    # Khởi tạo server và clients
    global_model = CNNModel()
    server = Server(global_model, workers, test_dataloader=test_dataloader)  # Truyền test_dataloader

    clients = [
        Client(
            client_id=i,
            model=CNNModel(),
            dataset=train_dataloader[i],
            config=CONFIG,
            worker=workers[i],
        )
        for i in range(CONFIG["num_clients"])
    ]

    print("Bắt đầu huấn luyện Federated Learning với FedAvg:")
    fedavg(clients, server, CONFIG["rounds"])


if __name__ == "__main__":
    main()
