import tkinter as tk
from tkinter import ttk


class CalculatorApp:
    """A modern desktop calculator built with Tkinter."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Calculator")
        self.root.geometry("380x560")
        self.root.minsize(320, 480)

        self.expression_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="0")

        self._configure_style()
        self._build_layout()

    def _configure_style(self) -> None:
        self.root.configure(bg="#0f172a")

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background="#0f172a")
        style.configure("Display.TFrame", background="#111827")
        style.configure(
            "Expression.TLabel",
            background="#111827",
            foreground="#9ca3af",
            font=("Segoe UI", 14),
            anchor="e",
            padding=(16, 16, 16, 0),
        )
        style.configure(
            "Result.TLabel",
            background="#111827",
            foreground="#f8fafc",
            font=("Segoe UI Semibold", 34),
            anchor="e",
            padding=(16, 0, 16, 18),
        )

        style.configure(
            "Calc.TButton",
            font=("Segoe UI Semibold", 16),
            padding=12,
            borderwidth=0,
            relief="flat",
            foreground="#e2e8f0",
            background="#1f2937",
        )
        style.map(
            "Calc.TButton",
            background=[("active", "#374151"), ("pressed", "#4b5563")],
            foreground=[("disabled", "#64748b")],
        )

        style.configure(
            "Operator.TButton",
            background="#2563eb",
            foreground="#ffffff",
        )
        style.map(
            "Operator.TButton",
            background=[("active", "#1d4ed8"), ("pressed", "#1e40af")],
        )

        style.configure(
            "Action.TButton",
            background="#334155",
            foreground="#f8fafc",
        )
        style.map(
            "Action.TButton",
            background=[("active", "#475569"), ("pressed", "#64748b")],
        )

        style.configure(
            "Equals.TButton",
            background="#059669",
            foreground="#ffffff",
        )
        style.map(
            "Equals.TButton",
            background=[("active", "#047857"), ("pressed", "#065f46")],
        )

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=16)
        container.pack(fill="both", expand=True)

        display = ttk.Frame(container, style="Display.TFrame")
        display.pack(fill="x", pady=(0, 14))

        expression_label = ttk.Label(
            display,
            textvariable=self.expression_var,
            style="Expression.TLabel",
        )
        expression_label.pack(fill="x")

        result_label = ttk.Label(
            display,
            textvariable=self.result_var,
            style="Result.TLabel",
        )
        result_label.pack(fill="x")

        grid = ttk.Frame(container, style="App.TFrame")
        grid.pack(fill="both", expand=True)

        buttons = [
            ["C", "⌫", "%", "/"],
            ["7", "8", "9", "*"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["±", "0", ".", "="],
        ]

        for row_index, row in enumerate(buttons):
            grid.rowconfigure(row_index, weight=1)
            for col_index, char in enumerate(row):
                grid.columnconfigure(col_index, weight=1)
                style = self._button_style(char)
                btn = ttk.Button(
                    grid,
                    text=char,
                    style=style,
                    command=lambda value=char: self._on_button_press(value),
                )
                btn.grid(
                    row=row_index,
                    column=col_index,
                    sticky="nsew",
                    padx=6,
                    pady=6,
                )

        self._bind_keyboard()

    @staticmethod
    def _button_style(value: str) -> str:
        if value in {"+", "-", "*", "/", "%"}:
            return "Operator.TButton"
        if value == "=":
            return "Equals.TButton"
        if value in {"C", "⌫", "±"}:
            return "Action.TButton"
        return "Calc.TButton"

    def _bind_keyboard(self) -> None:
        for key in "0123456789.+-*/%":
            self.root.bind(key, self._on_key_input)
        self.root.bind("<Return>", lambda _: self._evaluate())
        self.root.bind("<BackSpace>", lambda _: self._backspace())
        self.root.bind("<Escape>", lambda _: self._clear())

    def _on_key_input(self, event: tk.Event) -> None:
        self._append_to_expression(event.char)

    def _on_button_press(self, value: str) -> None:
        if value == "C":
            self._clear()
            return
        if value == "⌫":
            self._backspace()
            return
        if value == "=":
            self._evaluate()
            return
        if value == "±":
            self._toggle_sign()
            return

        self._append_to_expression(value)

    def _append_to_expression(self, value: str) -> None:
        expression = self.expression_var.get()
        expression += value
        self.expression_var.set(expression)

    def _clear(self) -> None:
        self.expression_var.set("")
        self.result_var.set("0")

    def _backspace(self) -> None:
        expression = self.expression_var.get()
        if expression:
            self.expression_var.set(expression[:-1])

    def _toggle_sign(self) -> None:
        expression = self.expression_var.get().strip()
        if not expression:
            return

        try:
            value = str(float(expression) * -1)
            if value.endswith(".0"):
                value = value[:-2]
            self.expression_var.set(value)
        except ValueError:
            self.result_var.set("Invalid")

    def _evaluate(self) -> None:
        expression = self.expression_var.get().strip()
        if not expression:
            return

        try:
            sanitized = expression.replace("%", "/100")
            result = eval(sanitized, {"__builtins__": None}, {})  # noqa: S307
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            self.result_var.set(str(result))
        except Exception:
            self.result_var.set("Error")


def main() -> None:
    root = tk.Tk()
    CalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
