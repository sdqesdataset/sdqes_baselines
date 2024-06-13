import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.video_decoding import decode_video

SIZE = 224
CENTERCROP = True
FRAMERATE = 5
DPI = 100

def draw_animation(clip_vid_rgb_frames, title="", framerate=FRAMERATE, save_path=None, save_dpi=DPI):
    n_frames = clip_vid_rgb_frames.shape[0]

    # actual plot
    fig, (ax, ax2) = plt.subplots(nrows=2)
    if title:
        ax.set_title(title, loc='center', wrap=True)
    ax2.set_xlim(0, n_frames-1)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.set_ylabel("Time", color="tab:blue", fontsize=14)

    # animation
    def animate_ith_step(i):
        # update image
        ax.imshow(clip_vid_rgb_frames[i,:,:,:])  # show ith frame
        # update moving line to correct time
        line.set_data([i, i], [0,1])
        return line, im

    # initial states for the animation (time line at 0 - first frame)
    line, = ax2.plot([],[], linestyle='--', color='black')
    im = ax.imshow(clip_vid_rgb_frames[0,:,:,:])  # show an initial one first

    # draw animation
    anim = animation.FuncAnimation(fig, animate_ith_step, frames=n_frames, interval=1000/framerate)

    # # overlay centercrop on top of full video to show what the model sees
    # ax.add_patch(plt.Rectangle(
    #     (
    #         int((clip_vid_rgb_frames.shape[2] - SIZE) / 2.0),        # x
    #         int((clip_vid_rgb_frames.shape[1] - SIZE) / 2.0),       # y
    #     ),
    #     SIZE, SIZE,
    #     linewidth=2, ec="white", fc="none")
    # )

    if save_path is not None:
        # save animation
        anim.save(save_path, dpi=save_dpi)
        # print(f"saved to {save_path}")
        plt.close(fig)
    else:
        return anim

def make_animation(vid_rgb_path, title="", size=SIZE, centercrop=CENTERCROP, framerate=FRAMERATE):
    """Generates animation of model predictions for a single row of the dataset. Animation includes decoded video.

    Args:
        model_scores (array): Model predictions for each frame.
        row (pandas.core.series.Series): Row of the dataset to make animation for.
        clip_endpoint (int): Index of the last frame (out of n_frames = len(model_scores) ) to include in the animation.
    """

    # decode full video
    vid_rgb_frames = decode_video(
        vid_rgb_path,
        size=size,
        centercrop=centercrop,
        framerate=framerate,
    )

    # draw animation
    anim = draw_animation(clip_vid_rgb_frames=vid_rgb_frames, title=title, framerate=framerate)

    return anim
