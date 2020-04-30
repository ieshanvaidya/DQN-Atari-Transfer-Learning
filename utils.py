import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



class Preprocessor:
    def __init__(self, agent_history_length=4, interpolation=Image.NEAREST):
        """
        Performs preprocessing steps on raw Atari input. Refer to DQN paper for terminology.
            agent_history_length: Number of frames to stack
            interpolation: Filter used during resize.
                           Refer to - https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
        """
        self.buffer = torch.zeros((agent_history_length, 84, 84))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=(84, 84), interpolation=interpolation),
            transforms.ToTensor()
        ])

        self.to_pil = transforms.ToPILImage()

    def reset(self):
        self.buffer = torch.zeros_like(self.buffer)

    def process(self, frame):
        """
        Process Atari frame and return tensor that can feed into the estimator
        """
        out = self.transform(frame)
        self.buffer[:-1] = self.buffer[1:].clone()
        self.buffer[-1] = out

        return self.buffer.clone()

    def visualize_buffer(self):
        n = self.buffer.shape[0]
        fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(12,24))
        for i in range(n):
            ax[i].imshow(self.to_pil(self.buffer[i]), cmap='gray')
            ax[i].axis('off')
        return fig



def frameskip(env, action, skips=4):
    """

    Performs frameskip. Applies same action over skipped frames.
    Max of last 2 frames is returned unless episode terminates.
    """
    all_reward, all_frames, all_done, all_info = 0, [], False, []

    for i in range(skips):
        frame, reward, done, info = env.step(action)
        all_frames.append(frame)
        all_reward += reward
        all_done |= done
        all_info.append(info)

        if done:
            break

    if len(all_frames) == skips:
        frame = np.maximum(all_frames[-2], all_frames[-1])

    else:
        frame = all_frames[-1]

    return frame, all_reward, all_done, all_info



def get_save_dir(base_dir, name, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')
