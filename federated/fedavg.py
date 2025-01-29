import time
import matplotlib.pyplot as plt
import torch
from config import CONFIG
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
    plt.savefig("img/result.png")

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
    
    # Trả về trạng thái mô hình đã tổng hợp
    return avg_state_dict 

    # Cập nhật mô hình toàn cục
    global_model.load_state_dict(avg_state_dict)


def fedavg(clients, server, rounds):
    """
    Thuật toán FedAvg với tính toán thời gian và ghi lại loss/accuracy.
    """
    loss_history = []
    accuracy_history = []

    start_time = time.time()  # Bắt đầu tính thời gian

    round_time = 0
    prev_loss = 0
    convergence = False
    while not convergence:
        print(f"\n--- Round {round_time} ---")
        # # Gửi mô hình đã được tách tới các client
        # for client in clients:
        #     client.set_model(server.global_model)  # Gửi bản sao của mô hình cho client

        # 1. Thu thập mô hình từ các client
        client_updates = []
        
        # Mô phỏng song song
        training_clients = np.random.choice(clients, CONFIG["num_client_training"], replace=False)
        with ThreadPoolExecutor() as executor:
            futures = {}
            # Gửi các tác vụ huấn luyện lên các thread, kèm theo client_id
            for client in training_clients:
                futures[executor.submit(client.train)] = client.client_id  # Lưu trữ future kèm theo client_id

            # Chờ kết quả và thu thập mô hình từ từng client
            for future in futures:
                client_model_state = future.result()  # Lấy kết quả huấn luyện từ client
                client_id = futures[future]  # Lấy client_id từ dictionary
                print(f"Client {client_id} finished training.")
                client_updates.append(client_model_state)

        # 2. Tổng hợp các mô hình trên server
        print("Aggregating updates on the server...")
        new_state_param_model = aggregate(server.global_model, client_updates)
        server.global_model.load_state_dict(new_state_param_model)

        # 3. Đánh giá trên server
        loss, accuracy = server.evaluate()
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

        # 4. Kiểm tra hội tụ
        if abs(loss-prev_loss) < CONFIG["convergence_threshold"] and loss < 0.5:
            convergence = True
            break
        prev_loss = loss
        round_time += 1

        # 5. Gui mô hình toàn cục cho các client
        with ThreadPoolExecutor() as executor:
            # Gửi các tác vụ huấn luyện lên các thread, kèm theo client_id
            for client in clients:
                executor.submit(client.set_model, server.global_model.state_dict())


    end_time = time.time()  # Kết thúc tính thời gian
    total_time = end_time - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # 5. Vẽ đồ thị loss và accuracy
    plot_metrics(loss_history, accuracy_history)
