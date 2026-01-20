import tkinter as tk
from tkinter import messagebox

# ==================================================
# EXERCISE 1 : Guessing Game (Binary Search)
# ==================================================

class BinaryGuessingGame:
    def __init__(self, n):
        self.min = 1
        self.max = n
        self.current_guess = None

    def make_guess(self):
        self.current_guess = (self.min + self.max) // 2
        return self.current_guess

    def guess_too_low(self):
        self.min = self.current_guess + 1

    def guess_too_high(self):
        self.max = self.current_guess - 1


class GuessingGameGUI:
    def __init__(self, root):
        self.root = root
        self.clear_window()
        self.root.title("Exercise 1 - Guessing Game")

        self.game = BinaryGuessingGame(100)

        tk.Label(root, text="Think of a number between 1 and 100").pack(pady=10)

        self.guess_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.guess_label.pack(pady=10)

        frame = tk.Frame(root)
        frame.pack()

        tk.Button(frame, text="Too Low", command=self.too_low, width=10).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Correct", command=self.correct, width=10).grid(row=0, column=1, padx=5)
        tk.Button(frame, text="Too High", command=self.too_high, width=10).grid(row=0, column=2, padx=5)

        self.new_guess()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def new_guess(self):
        self.guess_label.config(text=f"My guess: {self.game.make_guess()}")

    def too_low(self):
        self.game.guess_too_low()
        self.new_guess()

    def too_high(self):
        self.game.guess_too_high()
        self.new_guess()

    def correct(self):
        messagebox.showinfo("Success", f"Number found: {self.game.current_guess}")


# ==================================================
# EXERCISE 2 : Prime Checker (Binary Search)
# ==================================================
class BinarySearch:
    def __init__(self, array):
        self.array = array
        self.n = len(array)

    def search(self, target):
        min_index = 0
        max_index = self.n - 1

        while True:
            if max_index < min_index:
                return -1
            guess = (min_index + max_index) // 2

            if self.array[guess] == target:
                return guess

            elif self.array[guess] < target:
                min_index = guess + 1

            else:
                max_index = guess - 1

class PrimeSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise 2 - Prime Checker")

        self.primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67,
            71, 73, 79, 83, 89, 97
        ]

        self.searcher = BinarySearch(self.primes)

        tk.Label(root, text="Enter a number (0–100):").pack(pady=10)

        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)

        tk.Button(root, text="Search", command=self.check_prime).pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    def check_prime(self):
        try:
            target = int(self.entry.get())
            index = self.searcher.search(target)

            if index != -1:
                self.result_label.config(
                    text=f"{target} is PRIME (index {index})",
                    fg="green"
                )
            else:
                self.result_label.config(
                    text=f"{target} is NOT prime",
                    fg="red"
                )
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer")

# ==================================================
# MAIN IN TKINTER
# ==================================================

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary Search Exercises")

        tk.Label(
            root,
            text="Choose an Exercise",
            font=("Arial", 14, "bold")
        ).pack(pady=20)

        tk.Button(
            root,
            text="Exercise 1: Guessing Game",
            width=30,
            command=lambda: GuessingGameGUI(root)
        ).pack(pady=10)

        tk.Button(
            root,
            text="Exercise 2: Prime Checker",
            width=30,
            command=lambda: PrimeSearchGUI(root)
        ).pack(pady=10)


# ==================================================
# PROGRAM START
# ==================================================

if __name__ == "__main__":
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()
