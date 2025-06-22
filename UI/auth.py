
import json, os, hashlib, secrets
from typing import Optional, Dict
import tkinter as tk
from tkinter import messagebox

USERS_FILE = "users.json"
current_user: Optional[str] = None


def _hash_pw(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def _verify_pw(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return _hash_pw(password, salt) == stored


def _load_users() -> Dict[str, str]:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_user(name: str, pw_hash: str) -> None:
    users = _load_users()
    users[name] = pw_hash
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def register(username: str, password: str) -> None:
    """Dodaje użytkownika; rzuca ValueError jeśli już istnieje."""
    users = _load_users()
    if username in users:
        raise ValueError("Użytkownik już istnieje.")
    _save_user(username, _hash_pw(password))


def login(username: str, password: str) -> None:
    """Ustawia globalne current_user po poprawnym logowaniu."""
    global current_user
    users = _load_users()
    if username not in users:
        raise ValueError("Nie ma takiego użytkownika.")
    if not _verify_pw(password, users[username]):
        raise ValueError("Błędne hasło.")
    current_user = username


def login_dialog() -> Optional[str]:
    """Pokazuje okienko Tk i zwraca nazwę użytkownika albo None."""
    root = tk.Tk()
    root.title("Logowanie – MoodScan")
    root.resizable(False, False)
    root.minsize(200, 200)

    tk.Label(root, text="Nazwa użytkownika:").pack(padx=12, pady=(12, 2))
    user_ent = tk.Entry(root, width=25); user_ent.pack(padx=12)
    tk.Label(root, text="Hasło:").pack(padx=12, pady=(8, 2))
    pass_ent = tk.Entry(root, width=25, show="•"); pass_ent.pack(padx=12)
    user_ent.focus()

    result = {"user": None}

    def _do_login():
        try:
            login(user_ent.get().strip(), pass_ent.get())
        except ValueError as e:
            messagebox.showwarning("Błąd", str(e), parent=root); return
        result["user"] = user_ent.get().strip()
        root.destroy()

    def _do_register():
        try:
            register(user_ent.get().strip(), pass_ent.get())
            messagebox.showinfo("OK",
                                "Zarejestrowano – możesz się zalogować.",
                                parent=root)
        except ValueError as e:
            messagebox.showwarning("Błąd", str(e), parent=root)

    btn_fr = tk.Frame(root); btn_fr.pack(pady=10)
    tk.Button(btn_fr, text="Zaloguj", width=10,
              command=_do_login).pack(side="left", padx=4)
    tk.Button(btn_fr, text="Zarejestruj", width=10,
              command=_do_register).pack(side="left", padx=4)

    root.bind("<Return>", lambda *_: _do_login())
    root.bind("<Escape>", lambda *_: root.destroy())
    root.mainloop()
    return result["user"]
