import time
import matplotlib.pyplot as plt
import torch
from config import CONFIG


def plot_metrics(loss_history, accuracy_history):
    """Vẽ đồ thị loss và accuracy theo từng vòng huấn luyện."""
    rounds = range(1, len(loss_history) + 1)
    
    # Chuyển đổi tensor sang NumPy array
    loss_history = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in loss_history]
    accuracy_history = [acc.detach().numpy() if torch.is_tensor(acc) else acc for acc in accuracy_history]
    
    plt.figure(figsize=(12, 6))

    # Vẽ đồ thị loss
    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_history, label="Loss", marker="o")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    # Vẽ đồ thị accuracy
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracy_history, label="Accuracy", marker="o", color="green")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def aggregate(global_model, client_updates):
    """
    Tổng hợp các tham số mô hình từ các client bằng cách tính trung bình trọng số.
    
    Args:
        global_model: Mô hình toàn cục (torch.nn.Module).
        client_updates: Danh sách state_dict từ các client.
    
    Returns:
        state_dict: Trạng thái mô hình đã tổng hợp.
    """
    avg_state_dict = client_updates[0].copy()

    # Tính trung bình cho từng tham số trong mô hình
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.mean(
            torch.stack([client[key] for client in client_updates]), dim=0
        )

    # Cập nhật mô hình toàn cục
    global_model.load_state_dict(avg_state_dict)


def fedavg(clients, server, rounds):
    """
    Thuật toán FedAvg với tính toán thời gian và ghi lại loss/accuracy.
    """
    loss_history = []
    accuracy_history = []

    start_time = time.time()  # Bắt đầu tính thời gian

    # Huấn luyện cho đến khi hội tụ 
    round_time = 0
    prev_accuracy = 0
    convergence = False
    while not convergence:
        print(f"\n--- Round {round_time} ---")

        # 1. Thu thập mô hình từ các client
        client_updates = []
        for client in clients:
            print(f"Client {client.client_id} training...")
            client_model_state = client.train()  # Huấn luyện mô hình
            client_updates.append(client_model_state)

        # 2. Tổng hợp các mô hình trên server
        print("Aggregating updates on the server...")
        server.update_model(client_updates)

        # 3. Đánh giá trên server
        loss, accuracy = server.evaluate()
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

        # 4. Kiểm tra hội tụ
        if abs(accuracy-prev_accuracy) < CONFIG["convergence_threshold"]:
            convergence = True
            break
        prev_accuracy = accuracy
        round_time += 1
        
    # Huấn luyện theo số vòng lặp
    # for r in range(rounds):
    #     print(f"\n--- Round {r + 1}/{rounds} ---")

    #     # 1. Thu thập mô hình từ các client
    #     client_updates = []
    #     for client in clients:
    #         print(f"Client {client.client_id} training...")
    #         client_model_state = client.train()  # Huấn luyện mô hình
    #         client_updates.append(client_model_state)

    #     # 2. Tổng hợp các mô hình trên server
    #     print("Aggregating updates on the server...")
    #     server.update_model(client_updates)

    #     # 3. Đánh giá trên server
    #     loss, accuracy = server.evaluate()
    #     loss_history.append(loss)
    #     accuracy_history.append(accuracy)
    #     print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

    end_time = time.time()  # Kết thúc tính thời gian
    total_time = end_time - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # 4. Vẽ đồ thị loss và accuracy
    plot_metrics(loss_history, accuracy_history)

