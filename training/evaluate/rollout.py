"""Play legal moves for a batch of games or a single policy."""

from collections.abc import Callable, Sequence

from ..env import Game, encode, is_deadlocked, legal_mask, pushed_into_deadlock, step_game


def play_batch(
    games: Sequence[Game],
    choose_batch: Callable[[Sequence[bytes], Sequence[Sequence[int]]], Sequence[int]],
    max_steps: int,
    *,
    detect_cycle: bool = True,
    deadlock_detection: bool = True,
) -> list[str]:
    """Make one inference call for all games awaiting a decision in each round."""
    limits = [g.moves + max_steps for g in games]
    seen = [set() for _ in games]
    outcomes = ["死锁" if deadlock_detection and is_deadlocked(game) else "" for game in games]
    active = [i for i, outcome in enumerate(outcomes) if not outcome]
    while active:
        states = []
        masks = []
        pending = []
        for i in active:
            game = games[i]
            if game.won or game.moves >= limits[i]:
                outcomes[i] = "通关" if game.won else "达到步数上限"
                continue
            state = encode(game)
            if detect_cycle and state in seen[i]:
                outcomes[i] = "循环"
                continue
            mask = legal_mask(game)
            if not any(mask):
                outcomes[i] = "无可用动作"
                continue
            if detect_cycle:
                seen[i].add(state)
            states.append(state)
            masks.append(mask)
            pending.append(i)

        if not pending:
            break
        actions = choose_batch(states, masks)
        active = []
        for i, action in zip(pending, actions, strict=True):
            game = games[i]
            previous_boxes = game.boxes
            step_game(game, action)
            if deadlock_detection and pushed_into_deadlock(game, previous_boxes):
                outcomes[i] = "死锁"
            else:
                active.append(i)
    return outcomes


def play_level(
    game: Game,
    choose_action: Callable[[bytes, Sequence[int]], int],
    max_steps: int,
    *,
    detect_cycle: bool = True,
    deadlock_detection: bool = True,
) -> str:
    """Play up to max_steps primitive moves. Disable cycle detection for stochastic policies."""
    def choose_one(states, masks):
        return [choose_action(states[0], masks[0])]

    return play_batch(
        [game], choose_one, max_steps, detect_cycle=detect_cycle, deadlock_detection=deadlock_detection,
    )[0]
