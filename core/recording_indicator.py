"""Uzaktan görülebilir kayıt göstergesi — büyük renkli pencere.
Kırmızı = kayıt sürüyor (yerinde kal), Yeşil = kayıt bitti (hareket edebilirsin).
Terminal yazısı uzaktan okunamadığı için bu pencere fiziksel geri bildirim sağlar.
"""
import pygame

GRAY = (50, 50, 55)
RED = (220, 30, 30)
GREEN = (30, 200, 70)
WHITE = (255, 255, 255)


class RecordingIndicator:
    def __init__(self, size=520):
        pygame.init()
        self._screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Kayıt Göstergesi — uzaktan takip")
        self._font_big = pygame.font.SysFont(None, 120)
        self._font_small = pygame.font.SysFont(None, 38)
        self._color = GRAY
        self._text = ""
        self._sub = ""
        self._draw()

    def _draw(self):
        self._screen.fill(self._color)
        cx, cy = self._screen.get_width() // 2, self._screen.get_height() // 2
        if self._text:
            surf = self._font_big.render(self._text, True, WHITE)
            self._screen.blit(surf, surf.get_rect(center=(cx, cy - 35)))
        if self._sub:
            surf2 = self._font_small.render(self._sub, True, WHITE)
            self._screen.blit(surf2, surf2.get_rect(center=(cx, cy + 65)))
        pygame.display.flip()

    def pump(self):
        """Pencerenin donmaması için ana döngüden sık sık çağrılmalı.
        Ölçüm sürerken yanlışlıkla kapanmasın diye QUIT olayını yok sayar."""
        pygame.event.pump()
        pygame.event.get()

    def show_waiting(self, seconds_left):
        self._color, self._text, self._sub = GRAY, str(seconds_left), "HAZIRLAN..."
        self._draw()

    def show_recording(self, label, remaining):
        self._color, self._text = RED, "● KAYIT"
        self._sub = f"{label}  —  {remaining:.0f} sn kaldı  —  YERİNDE KAL"
        self._draw()

    def show_done(self):
        self._color, self._text, self._sub = GREEN, "✓ BİTTİ", "Hareket edebilirsin"
        self._draw()

    def close(self):
        pygame.quit()
