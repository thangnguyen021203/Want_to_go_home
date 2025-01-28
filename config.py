CONFIG = {
    "num_clients": 2,
    "batch_size": 32,
    "epochs": 3,
    "lr": 0.1,
    "dataset": "MNIST",
    "use_secret_sharing": True,  # Sử dụng PySyft để bảo mật
    "rounds": 2,
    "convergence_threshold": 0.01
}
