CONFIG = {
    "num_clients": 20,
    "batch_size": 32,
    "epochs": 3,
    "lr": 0.01,
    "dataset": "MNIST",
    "use_secret_sharing": True,  # Sử dụng PySyft để bảo mật
    "rounds": 2,
    "convergence_threshold": 0.01,
    "num_client_training": 2
}
