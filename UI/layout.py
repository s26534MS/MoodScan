import tkinter as tk


def build_gui(root):
    main = tk.Frame(root)
    main.pack(fill="both", expand=True)
    left = tk.Frame(main, width=580, height=500, bg="#d9d9d9")
    left.pack(side="left", padx=20, pady=20)
    left.pack_propagate(False)
    img_lbl = tk.Label(left, bg="#d9d9d9")
    img_lbl.pack(fill="both", expand=True)

    right = tk.Frame(main)
    right.pack(side="left", fill="both",
               expand=True, padx=(0, 20), pady=20)
    canvas = tk.Canvas(right, borderwidth=0, highlightthickness=0)
    vbar = tk.Scrollbar(right, orient="vertical", command=canvas.yview)
    resbox = tk.Frame(canvas)
    canvas.configure(yscrollcommand=vbar.set)
    canvas.create_window((0, 0), window=resbox, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    vbar.pack(side="right", fill="y")
    resbox.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    btnbar = tk.Frame(root, pady=8)
    btnbar.pack(side="bottom", fill="x")

    return {"img_lbl": img_lbl,
            "resbox": resbox,
            "canvas": canvas,
            "btnbar": btnbar,
            "left": left}
