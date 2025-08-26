"""Entry point for the application."""

from src.gui import App


def main():
    """Start the application."""
    import tkinter as tk

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
