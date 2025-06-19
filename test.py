import argparse
import torch
from src.tetris_env import Tetris3D, render_voxel_video_matplotlib
from src.deep_q_network import DeepQNetwork

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=4, help="Board width")
    parser.add_argument("--height", type=int, default=20, help="Board height")
    parser.add_argument("--depth", type=int, default=4, help="Board depth")
    parser.add_argument("--fps", type=int, default=20, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="./trained_3d_tetris_models/best_model.pth")
    parser.add_argument("--output", type=str, default="./output.gif")
    return parser.parse_args()


def test(opt, video_index=None):
    model = DeepQNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Load model
    checkpoint = torch.load(opt.saved_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 是 state_dict 格式的 checkpoint
        model = DeepQNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 是整個模型儲存的格式（torch.save(model)）
        model = checkpoint

    model = model.to(device).eval()

    env = Tetris3D(height=opt.height, width=opt.width, depth=opt.depth, device=device)
    env.reset()
    frames = []
    frame_rewards = []
    total_score = 0
    done = False

    while not done:
        # if total_score > 50:
        #     break

        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            predictions = model(next_states)[:, 0]

        epsilon = 0.001
        if torch.rand(1).item() < epsilon:
            index = torch.randint(len(next_actions), (1,)).item()
        else:
            index = torch.argmax(predictions).item()

        action = next_actions[index]

        prev_frame_len = len(frames)
        reward, done = env.step(action, render=False, frames=frames)
        new_frame_len = len(frames)
        num_new_frames = new_frame_len - prev_frame_len

        frame_rewards.extend([0.0] * (num_new_frames - 1))
        frame_rewards.append(reward)

        total_score += reward

    # video_name = opt.output 
    # render_voxel_video_matplotlib(
    #     board_sequence=frames,
    #     piece_colors=env.piece_colors,
    #     save_path=video_name,
    #     fps=opt.fps,
    #     frame_rewards=frame_rewards
    # )

    return total_score


if __name__ == "__main__":
    opt = get_args()
    scores = []
    for i in range(10):
        print(f"Running test game {i + 1}...")
        score = test(opt, video_index=i + 1)
        scores.append(score)
        print(f"Score: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\n🎮 Average Score over total runs: {avg_score:.2f}")
