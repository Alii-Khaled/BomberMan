import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)


def _progress_iter(n_rounds, gui):
    """Terminal-safe progress for long training runs.

    Jupyter xterm.js (+ tmux redraw) chokes on tqdm's default per-round
    \\r rewrites: a 1500-round run grows a single line to ~100KB and kills
    the terminal tab at random points (backend often survives in tmux).
    For headless training (gui is None, i.e. --no-gui) default to plain
    newline logs every 25 rounds. Opt in to a throttled bar with
    APEX_TQDM=1. Honors TQDM_DISABLE=1 / NO_TQDM=1. GUI runs keep tqdm.
    """
    if gui is not None:
        yield from tqdm(range(n_rounds))
        return
    if os.environ.get('TQDM_DISABLE', '') == '1' \
            or os.environ.get('NO_TQDM', '') == '1':
        yield from range(n_rounds)
        return
    if os.environ.get('APEX_TQDM', '0') != '1':
        every = int(os.environ.get('APEX_LOG_EVERY', '25') or 25)
        every = max(1, every)
        t0 = time()
        for i in range(n_rounds):
            yield i
            if (i + 1) % every == 0 or (i + 1) == n_rounds:
                dt = time() - t0
                print(f"[progress] round {i + 1}/{n_rounds} "
                      f"elapsed={dt:.0f}s", flush=True)
        return
    # Opt-in live bar: heavily throttled so the terminal sees ~1 update/min
    # instead of ~8/s; fixed width avoids xterm.js reflow blowups.
    yield from tqdm(range(n_rounds), mininterval=60.0, maxinterval=300.0,
                    miniters=25, dynamic_ncols=False, ncols=80,
                    file=sys.stderr)


def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in _progress_iter(n_rounds, gui):
        world.new_round()
        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]

            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None
            else:
                # Might want to wait
                pass

        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if gui is not None:
            do_continue = False
            while not do_continue:
                render(True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                            do_continue = True

    world.end()


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames
    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None
    world_controller(world, args.n_rounds,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()