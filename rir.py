import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Define your DQN class and functions
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize necessary variables and hyperparameters
state_size = 100  # Modify according to your input state dimensions
action_size = 2  # Modify according to your action space size

bounding_box_width = 128
image_width = 128
image_height = 72


# Define the reward function
def calculate_reward(bounding_box, frame_center):
    x1, _, x2, _ = bounding_box
    bounding_box_width = x2 - x1 + 1
    bounding_box_center = (x1 + x2) / 2

    if x1 > image_width * 0.41 and x2 < image_width * 0.43:
        # Positive reward when the bounding box is within the middle 50% of the image width
        distance = abs(bounding_box_center - frame_center)
        reward = distance / bounding_box_width
    elif x1 <= image_width * 0.32 or x2 >= image_width * 0.96:
        # Negative penalty when the bounding box is in the left 25% or right 25% of the image width
        reward = -1
    elif bounding_box_width > image_height * 0.5:
        # Negative penalty when the bounding box height occupies more than 50% of the image height
        reward = -1
    elif bounding_box_width < image_height * 0.1:
        # Negative penalty when the bounding box height takes less than 10% of the image height
        reward = -1
    else:
        # Default reward
        reward = 0

    return reward


# Load and preprocess your video dataset
video_dataset = '/Users/martinprabhu/Downloads/m201.mp4'
cap = cv2.VideoCapture(video_dataset)


def preprocess_frame(frame):
    # Resize the frame to a specific size
    new_width = 224
    new_height = 224
    resized_frame = cv2.resize(frame, (new_width, new_height))

    return resized_frame


# Initialize necessary variables and hyperparameters
state_size = 224 * 224 * 3  # Modify according to your input state dimensions
action_size = 2  # Modify according to your action space size

batch_size = 64
learning_rate = 0.001
num_episodes = 500
num_steps = 100
target_update_freq = 10

epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
gamma = 0.99

# Create an instance of the DQN class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(state_size, action_size).to(device)
target_dqn = DQN(state_size, action_size).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Initialize replay memory
replay_memory = []

# Start training the DQN network
for episode in range(num_episodes):
    cap = cv2.VideoCapture(video_dataset)
    done = False
    total_reward = 0

    for step in range(num_steps):
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)

        state = frame.flatten()  # Assuming the flattened frame is used as the state

        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = target_dqn(state_tensor)
            action = torch.argmax(q_values).item()

        # Perform the action and get the next state, reward, and done signal
        # Perform your action here and update the state, reward, and done signal accordingly
        next_state = frame.flatten()  # Update the next state

        # Perform object detection and get the bounding box coordinates
        # Replace this code with your actual object detection implementation
        # Example: Using OpenCV's Haar cascades
        face_cascade = cv2.CascadeClassifier('/Users/martinprabhu/Downloads/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Assume there is only one detected face
            (x, y, w, h) = faces[0]

            bounding_box = (x, y, x + w, y + h)  # Update the bounding box coordinates
            frame_center = image_width / 2  # Replace with the actual frame center

            reward = calculate_reward(bounding_box, frame_center)
            print("Reward:", reward)
        else:
            # No face detected, handle this case accordingly
            reward = 0  # Set a default reward value
            print("No  detected in the frame.")

        done = False  # Update the done signal

        # Store the transition in the replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Update the state
        state = next_state
        total_reward += reward

        # Perform the DQN network update
        if len(replay_memory) >= batch_size:
            batch = random.sample(replay_memory, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch_array = np.array(state_batch)
            state_batch_tensor = torch.FloatTensor(state_batch_array).to(device)
            action_batch_tensor = torch.LongTensor(action_batch).unsqueeze(1).to(device)
            reward_batch_tensor = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
            next_state_batch_array = np.array(next_state_batch)
            next_state_batch_tensor = torch.FloatTensor(next_state_batch_array).to(device)
            done_batch_tensor = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

            q_values = dqn(state_batch_tensor).gather(1, action_batch_tensor)
            target_q_values = target_dqn(next_state_batch_tensor).max(1)[0].unsqueeze(1).detach()
            target_q_values = reward_batch_tensor + gamma * target_q_values * (1 - done_batch_tensor)

            loss = criterion(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network
        if step % target_update_freq == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if done:
            break

    cap.release()  # Release the video capture

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Print episode statistics
    print("Episode: {}, Total Reward: {}, Epsilon: {:.4f}".format(episode, total_reward, epsilon))
