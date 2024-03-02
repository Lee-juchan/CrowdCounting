import flwr as fl

def main() -> None:
  # 전략 정의
  strategy = fl.server.strategy.FedAvg(
      fraction_fit=0.5,
      fraction_evaluate=0.5,
  )

  # Flower 서버 시작
  fl.server.start_server(
      server_address="0.0.0.0:8080",
      config=fl.server.ServerConfig(num_rounds=3),
      strategy=strategy,
  )

if __name__ == "__main__":
  main()
