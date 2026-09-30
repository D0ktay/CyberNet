"""Canlı konum tahmini göstergesi — uzaktan görülebilir büyük renkli pencere.
zone_A = kırmızı, zone_B = yeşil, empty = sarı. Üzerinde tahmin edilen sınıf
ve olasılık yüzdeleri büyük fontla yazılır — terminal yazısı uzaktan okunamadığı
için fiziksel/görsel geri bildirim sağlar.
"""
import pygame

RED = (215, 35, 35)
GREEN = (30, 195, 75)
YELLOW = (235, 200, 40)
GRAY = (60, 60, 65)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)

ZONE_COLOR = {"zone_A": RED, "zone_B": GREEN, "empty": YELLOW}
ZONE_TEXT = {"zone_A": WHITE, "zone_B": WHITE, "empty": DARK}


class LiveIndicator:
    def __init__(self, size=560):
        pygame.init()
        self._screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Canlı Konum Tahmini — uzaktan takip")
        self._font_big = pygame.font.SysFont(None, 130)
        self._font_mid = pygame.font.SysFont(None, 46)
        self._font_small = pygame.font.SysFont(None, 32)

    def pump(self):
        """Pencerenin donmaması için ana döngüden sık çağrılmalı."""
        pygame.event.pump()
        pygame.event.get()

    def show(self, best_label, labels, probs):
        color = ZONE_COLOR.get(best_label, GRAY)
        tcolor = ZONE_TEXT.get(best_label, WHITE)
        self._screen.fill(color)
        cx, w = self._screen.get_width() // 2, self._screen.get_width()

        big = self._font_big.render(best_label, True, tcolor)
        self._screen.blit(big, big.get_rect(center=(cx, 120)))

        y = 235
        for lbl, p in zip(labels, probs):
            bar_w = int((w - 80) * p)
            pygame.draw.rect(self._screen, tcolor, (40, y - 16, w - 80, 32), width=2)
            pygame.draw.rect(self._screen, tcolor, (40, y - 16, bar_w, 32))
            txt = self._font_mid.render(f"{lbl}  {p*100:4.1f}%", True,
                                        color if p > 0.5 else tcolor)
            self._screen.blit(txt, txt.get_rect(midleft=(52, y)))
            y += 56

        sub = self._font_small.render(">>> TAHMİN <<<", True, tcolor)
        self._screen.blit(sub, sub.get_rect(center=(cx, y + 24)))
        pygame.display.flip()

    def close(self):
        pygame.quit()
