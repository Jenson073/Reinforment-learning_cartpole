# 🤖 **Reinforcement Learning with Deep Q-Network (DQN)**  

This project demonstrates the implementation of a **Deep Q-Network (DQN)** to solve the classic control problem **CartPole-v1** (or **MountainCar-v0**) using the OpenAI Gym environment. The DQN agent learns optimal policies through experience replay and an epsilon-greedy strategy.  

---

## 🌟 **Project Overview**  

### **Environment**  
- **CartPole-v1**: A pole is attached to a cart, which moves along a frictionless track. The agent balances the pole by applying forces left or right.  
- **State Size**: `4` (position, velocity, angle, angular velocity).  
- **Action Size**: `2` (left or right).  

Alternatively, the code can be used for **MountainCar-v0**: The goal is to drive a car up a steep hill.  

---

## 🧠 **Deep Q-Network (DQN)**  

- **Model Architecture**:  
  - Input: State size (4).  
  - Two hidden layers with 24 neurons each and **ReLU** activation.  
  - Output: Action size (2) with **linear** activation for Q-values.  
  - Optimizer: **Adam** with learning rate `0.001`.  
  - Loss Function: **Mean Squared Error (MSE)**.  

- **Training Mechanism**:  
  - Experience replay: Stores past experiences in a replay memory buffer to break the correlation in sequential data.  
  - Target Q-Value:  
    \[
    Q_{\text{target}} = \text{reward} + \gamma \cdot \max(Q(\text{next state}))
    \]
  - Epsilon-greedy exploration: Balances exploration and exploitation with a decaying epsilon.  

---

## ⚙️ **Parameters**  

| Parameter           | Value              | Description                              |
|---------------------|--------------------|------------------------------------------|
| Episodes            | 20                | Number of episodes to train.            |
| Max Steps           | 100               | Maximum steps per episode.              |
| Learning Rate       | 0.001             | Learning rate for the optimizer.        |
| Discount Factor (γ) | 0.95              | Discount factor for future rewards.     |
| Epsilon             | 1.0 (decays)      | Initial exploration rate.               |
| Epsilon Decay       | 0.995             | Decay rate for epsilon.                 |
| Epsilon Min         | 0.01              | Minimum exploration rate.               |
| Batch Size          | 64                | Size of minibatch for training.         |
| Memory Size         | 2000              | Maximum size of replay memory.          |

---

## 📊 **Results**  

- The agent trains for 20 episodes, accumulating rewards over time.  
- Intermediate scores are printed every 10 steps for transparency.  
- A plot of scores over episodes visualizes the training progress.

---

## 🔧 **How to Run the Code**  

1. **Install Dependencies**:  
   Ensure you have Python installed along with the required libraries:  
   ```bash
   pip install gym tensorflow matplotlib numpy

---
