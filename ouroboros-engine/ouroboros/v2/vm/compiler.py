"""Compila um subconjunto imperativo de Python para a VM, sem exec/eval.

Suporta atribuições, condicionais, while, for range, arrays de entrada e return.
Imports, atributos e chamadas livres são rejeitados durante a compilação.
"""
from __future__ import annotations

import ast

from .language import Instruction, Program


class Compiler:
    def __init__(self):
        self.code = []
        self.locals = {}
        self.inputs = {}

    def emit(self, op, arg=None):
        self.code.append(Instruction(op, arg))
        return len(self.code) - 1

    def patch(self, address, target):
        self.code[address] = Instruction(self.code[address].op, target)

    def slot(self, name):
        if name not in self.locals:
            if len(self.locals) >= 16:
                raise ValueError("No máximo 16 variáveis locais")
            self.locals[name] = len(self.locals)
        return self.locals[name]

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            self.emit("PUSH", node.value)
        elif isinstance(node, ast.Name):
            if node.id in self.locals:
                self.emit("LOAD", self.locals[node.id])
            elif node.id in self.inputs:
                self.emit("INPUT", self.inputs[node.id])
            else:
                raise ValueError(f"Variável não definida: {node.id}")
        elif isinstance(node, ast.BinOp):
            operators = {ast.Add: "ADD", ast.Sub: "SUB", ast.Mult: "MUL", ast.Div: "DIV", ast.Mod: "MOD"}
            if type(node.op) not in operators:
                raise ValueError("Operador não suportado")
            self.expression(node.left)
            self.expression(node.right)
            self.emit(operators[type(node.op)])
        elif isinstance(node, ast.UnaryOp):
            self.expression(node.operand)
            if isinstance(node.op, ast.USub): self.emit("NEG")
            elif isinstance(node.op, ast.Not): self.emit("NOT")
            elif not isinstance(node.op, ast.UAdd): raise ValueError("Unário não permitido")
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            operators = {ast.Lt: "LT", ast.Gt: "GT", ast.LtE: "LE", ast.GtE: "GE", ast.Eq: "EQ"}
            if type(node.ops[0]) not in operators: raise ValueError("Comparação não suportada")
            self.expression(node.left)
            self.expression(node.comparators[0])
            self.emit(operators[type(node.ops[0])])
        elif isinstance(node, ast.BoolOp):
            self.expression(node.values[0])
            for value in node.values[1:]:
                self.expression(value)
                self.emit("AND" if isinstance(node.op, ast.And) else "OR")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "arr":
            self.expression(node.slice)
            self.emit("AGET")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == "arr":
                self.emit("LEN")
            elif name in {"abs", "min", "max"} and len(node.args) == (1 if name == "abs" else 2):
                for arg in node.args: self.expression(arg)
                self.emit(name.upper())
            else:
                raise ValueError("Chamada não permitida")
        else:
            raise ValueError(f"Expressão não permitida: {type(node).__name__}")

    def statements(self, statements):
        for node in statements:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.expression(node.value)
                self.emit("STORE", self.slot(node.targets[0].id))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                self.expression(ast.BinOp(left=node.target, op=node.op, right=node.value))
                self.emit("STORE", self.slot(node.target.id))
            elif isinstance(node, ast.Return) and node.value is not None:
                self.expression(node.value)
                self.emit("OUTPUT")
                self.emit("HALT")
            elif isinstance(node, ast.If):
                self.expression(node.test)
                conditional = self.emit("JZ", 0)
                self.statements(node.body)
                end = self.emit("JMP", 0)
                self.patch(conditional, len(self.code))
                self.statements(node.orelse)
                self.patch(end, len(self.code))
            elif isinstance(node, ast.While) and not node.orelse:
                start = len(self.code)
                self.expression(node.test)
                conditional = self.emit("JZ", 0)
                self.statements(node.body)
                self.emit("JMP", start)
                self.patch(conditional, len(self.code))
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                iterator = node.iter
                if not (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                        and iterator.func.id == "range" and not iterator.keywords and 1 <= len(iterator.args) <= 2):
                    raise ValueError("Somente for em range(stop) ou range(start, stop)")
                args = iterator.args
                self.expression(args[0] if len(args) == 2 else ast.Constant(value=0))
                variable = self.slot(node.target.id)
                self.emit("STORE", variable)
                end_slot = self.slot(f"__end_{variable}_{len(self.code)}")
                self.expression(args[-1])
                self.emit("STORE", end_slot)
                start = len(self.code)
                self.emit("LOAD", variable)
                self.emit("LOAD", end_slot)
                self.emit("LT")
                conditional = self.emit("JZ", 0)
                self.statements(node.body)
                self.emit("LOAD", variable)
                self.emit("PUSH", 1)
                self.emit("ADD")
                self.emit("STORE", variable)
                self.emit("JMP", start)
                self.patch(conditional, len(self.code))
            elif isinstance(node, ast.Pass):
                self.emit("NOP")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass  # Docstring.
            else:
                raise ValueError(f"Comando não permitido: {type(node).__name__}")


def compile_source(source: str, name="candidate") -> Program:
    if not isinstance(source, str) or len(source) > 32_768:
        raise ValueError("Fonte ausente ou grande demais")
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 2048:
        raise ValueError("AST grande demais")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Defina uma única função solve com parâmetros escalares e arr opcional")
    function = tree.body[0]
    args = function.args
    if (function.name != "solve" or function.decorator_list or args.vararg or args.kwarg
            or args.kwonlyargs or args.defaults or args.posonlyargs):
        raise ValueError("Assinatura não suportada")
    compiler = Compiler()
    scalars = [arg.arg for arg in args.args if arg.arg != "arr"]
    if len(scalars) > 8 or len(set(arg.arg for arg in args.args)) != len(args.args):
        raise ValueError("Parâmetros inválidos")
    compiler.inputs = {arg: i for i, arg in enumerate(scalars)}
    compiler.statements(function.body)
    compiler.emit("HALT")  # Destino válido para saltos que saem do último bloco.
    return Program(tuple(compiler.code), name).validate()
