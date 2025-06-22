import tkinter as tk
from tkinter import filedialog, messagebox
import cv2, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic import predictors, image_utils, history
import logic.stats as st


def attach(root: tk.Tk, widgets: dict, current_user: str) -> None:
    """Łączy przyciski i logikę z przygotowanym layoutem."""
    img_lbl = widgets["img_lbl"]
    resbox = widgets["resbox"]
    btnbar = widgets["btnbar"]
    canvas = widgets["canvas"]

    select_var = tk.StringVar(value="")
    radio_bar = tk.Frame(root, bg="#ececec", pady=4)
    radio_bar.pack(side="bottom", fill="x", before=btnbar)

    tk.Label(radio_bar, text="Prawdziwa emocja:",
             bg="#ececec", font=("Arial", 11, "bold")).pack(side="left", padx=6)

    for emo in predictors.OWN_CLASSES:
        tk.Radiobutton(radio_bar, text=emo.capitalize(),
                       variable=select_var, value=emo,
                       bg="#ececec").pack(side="left", padx=4)

    prediction_rows: list[tuple[str, str]] = []
    current_path = [""]

    def confirm_label() -> None:
        if not current_path[0]:
            messagebox.showwarning("Brak zdjęcia",
                                   "Najpierw wybierz lub zrób zdjęcie.")
            return
        if not select_var.get():
            messagebox.showwarning("Brak wyboru",
                                   "Zaznacz prawdziwą emocję.")
            return

        user_emo = select_var.get().lower()
        ok_cnt = 0
        for model, pred in prediction_rows:
            correct = pred.lower() == user_emo
            if history.update_label(current_path[0], model,
                                    user_emo, current_user, correct):
                ok_cnt += 1
        _update_stats_btn()
        messagebox.showinfo("Zapisano",
                            f"Zaktualizowano {ok_cnt} wierszy – trafne modele oznaczone.")

    tk.Button(radio_bar, text="Zatwierdź etykietę",
              command=confirm_label).pack(side="right", padx=8)

    def show(path: str) -> None:
        current_path[0] = path
        select_var.set("")
        prediction_rows.clear()
        for w in resbox.winfo_children(): w.destroy()

        photo = image_utils.load_photo(path)
        img_lbl.config(image=photo)
        img_lbl.image = photo

        # DeepFace
        emo_df, dist_df = predictors.predict_deepface(path)
        _block(path, "DeepFace", emo_df, dist_df)

        # CNN-y
        for lbl in predictors.MODEL_PATHS:
            emo, dist = predictors.predict_cnn(path, lbl)
            _block(path, lbl, emo, dist)

        root.update_idletasks()
        need_h = max(img_lbl.winfo_height(),
                     resbox.winfo_reqheight()) + 160
        root.geometry(f"{root.winfo_width()}x{min(950, need_h)}")

    def _block(path: str, model: str, emo: str, dist) -> None:
        prediction_rows.append((model, emo))

        tk.Label(resbox, text=f"[{model}] {emo.capitalize()}",
                 font=("Arial", 16, "bold")).pack(anchor="w", pady=(10, 2))

        if isinstance(dist, dict):  # DeepFace
            for e, v in dist.items():
                tk.Label(resbox,
                         text=f"• {e.capitalize():<7} – {v:5.1f} %",
                         font=("Arial", 12)).pack(anchor="w")
        else:  # CNN
            for i, v in enumerate(dist):
                tk.Label(resbox,
                         text=f"• {predictors.OWN_CLASSES[i].capitalize():<7} – {v * 100:5.1f} %",
                         font=("Arial", 12)).pack(anchor="w")

        history.save_row(path, model, emo, current_user)
        _update_stats_btn()

    def choose_file():
        f = filedialog.askopenfilename(title="Wybierz zdjęcie",
                                       filetypes=[("Pliki graficzne",
                                                   "*.jpg *.jpeg *.png")])
        if f: show(f)

    def take_photo():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Błąd", "Brak dostępu do kamerki.");
            return

        cv2.namedWindow("Kamerka Spacja = zapis   Esc = anuluj")
        while True:
            ok, frame = cap.read()
            if not ok: break

            cv2.putText(frame, "SPACJA = zapis   ESC = anuluj",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Kamerka Spacja = zapis   Esc = anuluj", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if k == 32:
                cv2.imwrite("snapshot.jpg", frame)
                cap.release()
                cv2.destroyAllWindows()
                show("snapshot.jpg")
                return

    def clear_all():
        current_path[0] = ""
        select_var.set("")
        img_lbl.config(image="")
        img_lbl.image = None
        for w in resbox.winfo_children(): w.destroy()
        canvas.yview_moveto(0)
        root.geometry("1000x650")

    def show_stats():
        df_all = history.load_stats(current_user)
        if df_all is None or df_all.empty:
            messagebox.showinfo("Brak danych",
                                "Brak rekordów w pliku historii.")
            return

        acc = st.model_accuracy(df_all) * 100
        fig1, ax1 = plt.subplots(figsize=(5.5, 4))
        ax1.bar(acc.index, acc.values, color="#1f77b4")
        ax1.set_title("Accuracy modeli (%)")
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=15)

        cm, lbls = st.confusion(df_all)
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        if cm is None:
            ax2.text(0.5, 0.5, "Brak pełnych etykiet",
                     ha="center", va="center", fontsize=12)
            ax2.axis("off")
        else:
            im = ax2.imshow(cm, cmap="Blues")
            ax2.set_xticks(range(len(lbls)))
            ax2.set_xticklabels(lbls, rotation=45, ha="right")
            ax2.set_yticks(range(len(lbls)))
            ax2.set_yticklabels(lbls)
            ax2.set_title("Confusion matrix")
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    ax2.text(j, i, cm[i, j],
                             ha="center", va="center", color="black")

        win = tk.Toplevel()
        win.title("Statystyki")
        win.geometry("1200x650")
        win.minsize(1000, 550)
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)

        FigureCanvasTkAgg(fig1, master=frame
                          ).get_tk_widget().pack(side="left",
                                                 fill="both", expand=True)
        FigureCanvasTkAgg(fig2, master=frame
                          ).get_tk_widget().pack(side="left",
                                                 fill="both", expand=True)

    def _update_stats_btn():
        df = history.load_stats(current_user)
        state = "normal" if df is not None and not df.empty else "disabled"
        stats_btn.config(state=state)

    def show_mood():
        df = history.load_stats(current_user)
        if df is None or df.empty:
            messagebox.showinfo("Brak danych",
                                "Brak rekordów w pliku historii.")
            return

        df["date"] = pd.to_datetime(df["czas"]).dt.normalize()
        mood = df.groupby("date")["score"].mean()
        mood.index = pd.to_datetime(mood.index)

        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mood.index, mood.values, marker="o")
        ax.set_ylim(-1, 1)
        ax.set_title("Średni nastrój dziennie  (-1 = negatywny … +1 = happy)")
        ax.set_xlabel("Data")
        ax.set_ylabel("Score")
        ax.axhline(0, color="grey", lw=0.7)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        win = tk.Toplevel()
        win.title("Nastrój w czasie")
        FigureCanvasTkAgg(fig, master=win
                          ).get_tk_widget().pack(fill="both", expand=True)

    tk.Button(btnbar, text="Wybierz zdjęcie",
              command=choose_file).pack(side="left", padx=6)
    tk.Button(btnbar, text="Zrób zdjęcie",
              command=take_photo).pack(side="left", padx=6)
    tk.Button(btnbar, text="Wyczyść",
              command=clear_all).pack(side="left", padx=6)
    stats_btn = tk.Button(btnbar, text="Statystyki",
                          command=show_stats)
    _update_stats_btn()
    stats_btn.pack(side="left", padx=6)
    tk.Button(btnbar, text="Nastrój w czasie",
              command=show_mood).pack(side="left", padx=6)
