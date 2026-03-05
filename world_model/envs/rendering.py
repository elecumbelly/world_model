"""RGB rendering utilities and Pygame interactive viewer."""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class PygameViewer:
    """Interactive Pygame window for viewing/controlling the grid world."""

    def __init__(self, width: int = 512, height: int = 512, title: str = "Grid World"):
        if not HAS_PYGAME:
            raise RuntimeError("pygame is required for interactive viewing")
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)

    def render_frame(self, obs: np.ndarray, info_text: str = ""):
        """Display an observation frame. obs: (H, W, 3) uint8."""
        # Scale up to window size
        surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        surface = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))

        if info_text:
            text_surface = self.font.render(info_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    def get_action(self) -> int | None:
        """Wait for arrow key press, return action (0-3) or None to quit."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        return 0
                    elif event.key == pygame.K_DOWN:
                        return 1
                    elif event.key == pygame.K_LEFT:
                        return 2
                    elif event.key == pygame.K_RIGHT:
                        return 3
                    elif event.key == pygame.K_ESCAPE:
                        return None
            self.clock.tick(30)

    def close(self):
        pygame.quit()


class DualPaneViewer:
    """Side-by-side viewer: real world (left) vs. model prediction (right)."""

    def __init__(self, pane_size: int = 384, title: str = "Real vs Dream"):
        if not HAS_PYGAME:
            raise RuntimeError("pygame is required for interactive viewing")
        pygame.init()
        self.pane_size = pane_size
        gap = 20
        self.width = pane_size * 2 + gap
        self.height = pane_size + 40  # Extra for text
        self.gap = gap
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)

    def render_pair(self, real_obs: np.ndarray, dream_obs: np.ndarray,
                    step: int = 0, reward_real: float = 0, reward_dream: float = 0):
        """Show real (left) and dream (right) side by side."""
        self.screen.fill((30, 30, 30))

        for i, (obs, label, reward) in enumerate([
            (real_obs, "REAL", reward_real),
            (dream_obs, "DREAM", reward_dream),
        ]):
            surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
            surface = pygame.transform.scale(surface, (self.pane_size, self.pane_size))
            x = i * (self.pane_size + self.gap)
            self.screen.blit(surface, (x, 30))
            text = self.font.render(f"{label} | Step {step} | R={reward:.2f}", True, (255, 255, 255))
            self.screen.blit(text, (x + 10, 5))

        pygame.display.flip()
        self.clock.tick(10)

    def get_action(self) -> int | None:
        """Same as PygameViewer.get_action."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        return 0
                    elif event.key == pygame.K_DOWN:
                        return 1
                    elif event.key == pygame.K_LEFT:
                        return 2
                    elif event.key == pygame.K_RIGHT:
                        return 3
                    elif event.key == pygame.K_ESCAPE:
                        return None
            self.clock.tick(30)

    def close(self):
        pygame.quit()
