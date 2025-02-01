import pandas as pd
import matplotlib.pyplot as plt

# Load logged data
df = pd.read_csv("logs/training_log_20.csv")

# Plot Episode Reward Curve
plt.figure(figsize=(10, 5))
plt.plot(df["Timesteps"], df["EpisodeReward"], label="Episode Reward", color='blue')
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO Learning Curve")
plt.legend()
plt.savefig('./training_graphs/Learning_curve.png')

# Plot Losses
plt.figure(figsize=(10, 5))
plt.plot(df["Timesteps"], df["PolicyLoss"], label="Policy Loss", color='red')
plt.plot(df["Timesteps"], df["ValueLoss"], label="Value Loss", color='green')
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("PPO Losses")
plt.legend()
plt.savefig('./training_graphs/Losses.png')

# Plot Entropy and KL Divergence
plt.figure(figsize=(10, 5))
plt.plot(df["Timesteps"], df["EntropyLoss"], label="Entropy Loss", color='orange')
plt.plot(df["Timesteps"], df["KL_Divergence"], label="KL Divergence", color='purple')
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("PPO Entropy & KL Divergence")
plt.legend()
plt.savefig('./training_graphs/Entropy_KLdivergence.png')
