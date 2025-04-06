import torch
import torchvision.transforms as transforms
import numpy as np
import csv
import os
import argparse

import environment
from agent import Agent
from sac_agent import SACAgent

parser = argparse.ArgumentParser(description='Train RL agent on pushing task')
parser.add_argument('--algorithm', type=str, default='vpg', choices=['vpg', 'sac'],
                    help='RL algorithm to use (vpg or sac)')
parser.add_argument('--episodes', type=int, default=5000,
                    help='Number of episodes to train')
parser.add_argument('--render', action='store_true',
                    help='Render environment during training')
parser.add_argument('--test', action='store_true',
                   help='Test a trained model')
parser.add_argument('model_path', type=str, nargs='?', default="logs/vpg/model_final.pt",
                   help='Path to trained model (positional argument)')
args = parser.parse_args()

class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.075  # easier goal detection
        self._max_timesteps = 300  
        self._prev_obj_pos = None  # track object movement

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene
    
    def reset(self):
        super().reset()
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()  # initialize previous position
        self._t = 0

        try:
            return self.high_level_state()
        except:
            return None

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def vpg_reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        r_ee_to_obj = -0.1 * d_ee_to_obj  # Encourage moving towards object
        r_obj_to_goal = -0.5 * d_obj_to_goal  # Stronger reward for moving object to goal
        
        # Terminal bonus
        r_terminal = 20.0 if self.is_terminal() else 0.0
        
        # Step penalty
        r_step = -0.05
        
        # Movement bonus
        obj_movement = np.linalg.norm(obj_pos - self._prev_obj_pos)
        r_movement = 0.2 * obj_movement if obj_movement > 0.01 else 0.0
        
        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_terminal + r_step + r_movement

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based 
        r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal

        # direction bonus
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  #  division by zero
            r_direction = 0.0

        # terminal bonus
        r_terminal = 10.0 if self.is_terminal() else 0.0

        r_step = -0.1  # penalty for each step

        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.clamp(-1, 1).cpu().numpy() * self._delta
        else:
            action = np.clip(action, -1, 1) * self._delta
        
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)            

        self._t += 1

        state = self.high_level_state()
        # Use different reward functions based on algorithm
        if args.algorithm == 'vpg':
            reward = self.vpg_reward()
        else:
            reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated() if result else True
        return state, reward, terminal, truncated


    # def step(self, action):
    #     action = action.clamp(-1, 1).cpu().numpy() * self._delta
    #     ee_pos = self.data.site(self._ee_site).xpos[:2]
    #     target_pos = np.concatenate([ee_pos, [1.06]])
    #     target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
    #     self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
    #     self._t += 1

    #     state = self.high_level_state()
    #     reward = self.reward()
    #     terminal = self.is_terminal()
    #     truncated = self.is_truncated()
    #     return state, reward, terminal, truncated


def test_model(model_path, num_episodes=100, render=False, algorithm='vpg'):
    render_mode = "gui" if render else "offscreen"
    env = Hw3Env(render_mode=render_mode)
    
    if algorithm == 'vpg':
        agent = Agent()
     
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()  
    else:  
        agent = SACAgent()
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.actor.eval()
        agent.critic.eval()
    
    total_rewards = []
    success_count = 0
    
    for i in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done:
            with torch.no_grad():
                action = agent.decide_action(state)
            
            next_state, reward, is_terminal, is_truncated = env.step(action)
            
            cumulative_reward += reward
            done = is_terminal or is_truncated
            state = next_state
            episode_steps += 1

        if is_terminal:
            success_count += 1
            
        total_rewards.append(cumulative_reward)
        print(f"Test Episode={i}, reward={cumulative_reward}, steps={episode_steps}, success={is_terminal}")

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    print("\nTest Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Total episodes: {num_episodes}")
    
    return avg_reward, success_rate

if __name__ == "__main__":
    if args.test:
        algorithm = 'sac' if 'sac' in args.model_path else 'vpg'
        test_model(args.model_path, num_episodes=5, render=args.render, algorithm=algorithm)
    else:
        render_mode = "gui" if args.render else "offscreen"
        env = Hw3Env(render_mode=render_mode)
        
        if args.algorithm == 'vpg':
            agent = Agent()
            log_dir = "logs/vpg"
        else:  
            agent = SACAgent()
            log_dir = "logs/sac"
        
        num_episodes = args.episodes
        
        os.makedirs(log_dir, exist_ok=True)
        
        with open(f"{log_dir}/training_rewards.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Episode", "Reward"])
            
            if args.algorithm == 'vpg':
                # Use ReduceLROnPlateau which decreases LR when performance plateaus
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    agent.optimizer, 
                    mode='max',  
                    factor=0.5,  
                    patience=50, 
                    verbose=True
                )
            
            for i in range(num_episodes):
                env.reset()
                state = env.high_level_state()
                done = False
                cumulative_reward = 0.0
                episode_steps = 0

                while not done:
                    action = agent.decide_action(state)
                    next_state, reward, is_terminal, is_truncated = env.step(action)
                    
                    agent.store_transition(state, action, reward, next_state, is_terminal or is_truncated)
                    
                    agent.update_model()
                    
                    cumulative_reward += reward
                    done = is_terminal or is_truncated
                    state = next_state
                    episode_steps += 1

                print(f"Episode={i}, algorithm={args.algorithm}, reward={cumulative_reward}, steps={episode_steps}")
                
                csv_writer.writerow([i, cumulative_reward])
                csvfile.flush()
                
                if args.algorithm == 'vpg':
                    scheduler.step(cumulative_reward)
                    if i % 10 == 0:
                        print(f"Episode {i}: Current learning rate is {agent.optimizer.param_groups[0]['lr']}")

                if i % 100 == 0:
                    if args.algorithm == 'vpg':
                        torch.save(agent.model.state_dict(), f"{log_dir}/model_checkpoint_{i}.pt")
                    else:
                        torch.save({
                            'actor': agent.actor.state_dict(),
                            'critic': agent.critic.state_dict(),
                        }, f"{log_dir}/model_checkpoint_{i}.pt")

        if args.algorithm == 'vpg':
            torch.save(agent.model.state_dict(), f"{log_dir}/model_final.pt")
        else:
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
            }, f"{log_dir}/model_final.pt")
        

