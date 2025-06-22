import sys, cv2, tkinter as tk
from UI.layout import build_gui
from UI.controls import attach
from UI.auth import login_dialog

user = login_dialog()
if user is None:
    sys.exit("Zamknięto okno logowania – koniec programu.")

root = tk.Tk()
root.title(f"MoodScan Pro  –  {user}")
root.geometry("1000x650")
root.minsize(1000, 600)

widgets = build_gui(root)
attach(root, widgets, current_user=user)


def _on_close():
    """
    Zamyka okno Tk + wszystkie okna OpenCV
    i kończy proces (gdyby coś jeszcze wisiało).
    """
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    root.quit()
    root.destroy()
    sys.exit(0)


root.protocol("WM_DELETE_WINDOW", _on_close)

root.mainloop()
