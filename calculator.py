from __future__ import annotations

import argparse
import ast
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Professional Calculator</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --text: #f8fafc;
      --muted: #94a3b8;
      --btn: #1f2937;
      --btn-hover: #334155;
      --operator: #2563eb;
      --operator-hover: #1d4ed8;
      --equals: #059669;
      --equals-hover: #047857;
      --action: #475569;
      --radius: 16px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: radial-gradient(circle at top, #1e293b 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }

    .calculator {
      width: min(92vw, 390px);
      background: var(--panel);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }

    .display {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: var(--radius);
      padding: 14px;
      margin-bottom: 14px;
      min-height: 108px;
      text-align: right;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      gap: 8px;
      overflow: hidden;
    }

    .expression {
      color: var(--muted);
      font-size: 1rem;
      min-height: 1.2rem;
      word-break: break-all;
    }

    .result {
      font-size: clamp(2rem, 4.8vw, 2.4rem);
      font-weight: 600;
      word-break: break-all;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 14px;
      padding: 16px 0;
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--text);
      background: var(--btn);
      cursor: pointer;
      transition: 140ms ease;
    }

    button:hover { background: var(--btn-hover); }
    button:active { transform: translateY(1px) scale(0.99); }

    .operator { background: var(--operator); }
    .operator:hover { background: var(--operator-hover); }

    .equals { background: var(--equals); }
    .equals:hover { background: var(--equals-hover); }

    .action { background: var(--action); }

    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.82rem;
      text-align: center;
    }
  </style>
</head>
<body>
  <main class="calculator" aria-label="Calculator">
    <section class="display" aria-live="polite">
      <div id="expression" class="expression"></div>
      <div id="result" class="result">0</div>
    </section>

    <section class="grid">
      <button class="action" data-value="C">C</button>
      <button class="action" data-value="BACK">⌫</button>
      <button class="operator" data-value="%">%</button>
      <button class="operator" data-value="/">÷</button>

      <button data-value="7">7</button>
      <button data-value="8">8</button>
      <button data-value="9">9</button>
      <button class="operator" data-value="*">×</button>

      <button data-value="4">4</button>
      <button data-value="5">5</button>
      <button data-value="6">6</button>
      <button class="operator" data-value="-">−</button>

      <button data-value="1">1</button>
      <button data-value="2">2</button>
      <button data-value="3">3</button>
      <button class="operator" data-value="+">+</button>

      <button class="action" data-value="SIGN">±</button>
      <button data-value="0">0</button>
      <button data-value=".">.</button>
      <button class="equals" data-value="=">=</button>
    </section>

    <p class="hint">Keyboard: 0-9 + - * / % . Enter Backspace Esc</p>
  </main>

  <script>
    const expressionEl = document.getElementById("expression");
    const resultEl = document.getElementById("result");

    let expression = "";

    const setExpression = (next) => {
      expression = next;
      expressionEl.textContent = expression;
      if (!expression) resultEl.textContent = "0";
    };

    const calculate = async () => {
      if (!expression.trim()) return;
      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expression })
        });
        const payload = await response.json();
        resultEl.textContent = payload.result;
      } catch {
        resultEl.textContent = "Error";
      }
    };

    const toggleSign = () => {
      if (!expression.trim()) return;
      if (/^-?\d*\.?\d+$/.test(expression.trim())) {
        expression = String(parseFloat(expression) * -1);
        if (expression.endsWith(".0")) expression = expression.slice(0, -2);
        expressionEl.textContent = expression;
      } else {
        resultEl.textContent = "Invalid";
      }
    };

    const onInput = (value) => {
      if (value === "C") return setExpression("");
      if (value === "BACK") return setExpression(expression.slice(0, -1));
      if (value === "=") return calculate();
      if (value === "SIGN") return toggleSign();
      setExpression(expression + value);
    };

    document.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => onInput(btn.dataset.value));
    });

    document.addEventListener("keydown", (event) => {
      const key = event.key;
      if (/^[0-9.+\-*/%]$/.test(key)) {
        onInput(key);
      } else if (key === "Enter") {
        onInput("=");
      } else if (key === "Backspace") {
        onInput("BACK");
      } else if (key === "Escape") {
        onInput("C");
      }
    });
  </script>
</body>
</html>
"""


_ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], callable] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}

_ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], callable] = {
    ast.UAdd: lambda operand: +operand,
    ast.USub: lambda operand: -operand,
}


class ExpressionError(ValueError):
    """Raised when an expression contains unsupported syntax."""


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        operand = _safe_eval_node(node.operand)
        return _ALLOWED_UNARY_OPERATORS[type(node.op)](operand)

    raise ExpressionError("Unsupported expression")


def evaluate_expression(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        return "0"

    sanitized = expression.replace("%", "/100")
    try:
        tree = ast.parse(sanitized, mode="eval")
        result = _safe_eval_node(tree)
    except Exception:
        return "Error"

    if result.is_integer():
        return str(int(result))
    return str(result)


class CalculatorHandler(BaseHTTPRequestHandler):
    def _send(self, content: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(b"Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/evaluate":
            self._send(b"Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            expression = str(payload.get("expression", ""))
            result = evaluate_expression(expression)
            body = json.dumps({"result": result}).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        except Exception:
            self._send(
                json.dumps({"result": "Error"}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def _run_self_test() -> None:
    test_cases = {
        "2+2": "4",
        "5*(3+1)": "20",
        "50%": "0.5",
        "-3+10/2": "2",
        "2**8": "Error",
        "hello": "Error",
        "10/0": "Error",
    }
    for expression, expected in test_cases.items():
        actual = evaluate_expression(expression)
        assert actual == expected, f"{expression!r}: expected {expected}, got {actual}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone web calculator")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--self-test", action="store_true", help="Run internal expression tests and exit")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        print("Self-test passed.")
        return

    with ThreadingHTTPServer((args.host, args.port), CalculatorHandler) as server:
        print("Starting calculator web app (no tkinter required).")
        print(f"Open: http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
