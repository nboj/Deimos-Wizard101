from typing import Optional
from enum import Enum, auto

from .tokenizer import Tokenizer, Token
from .parser import *
from .types import *


class SemError(Exception):
    pass


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self._syms: list[Symbol] = []
        self._unique_player_selectors: set[PlayerSelector] = set()

    # Only blocks require lookup for now. Variables are only used internally and the sym is always locally known
    def lookup_block_by_name(self, literal: str) -> Symbol:
        for sym in reversed(self._syms):
            if sym.kind != SymbolKind.block:
                continue
            if sym.literal == literal:
                return sym
        if self.parent is not None:
            return self.parent.lookup_block_by_name(literal)
        raise SemError(f"Unable to find symbol in scope: {literal}")

    def put_sym(self, sym: Symbol) -> Symbol:
        self._syms.append(sym)
        return sym


class Analyzer:
    def __init__(self, stmts: list[Stmt]):
        self.scope = Scope()
        self._next_sym_id = 0
        self._block_defs: list[BlockDefStmt] = []
        self._stmts = stmts

        self._block_nesting_level = 0
        self._loop_nesting_level = 0

    def open_block(self):
        self.open_scope()
        self._block_nesting_level+=1

    def close_block(self):
        self.close_scope()
        self._block_nesting_level-=1

    def open_loop(self):
        self.open_scope()
        self._loop_nesting_level+=1

    def close_loop(self):
        self.close_scope()
        self._loop_nesting_level-=1

    def open_scope(self):
        self.scope = Scope(self.scope)

    def close_scope(self):
        self.scope = self.scope.parent

    def gen_sym_id(self) -> int:
        result = self._next_sym_id
        self._next_sym_id += 1
        return result

    def gen_block_sym(self, name: str) -> Symbol:
        return self.scope.put_sym(Symbol(name, self.gen_sym_id(), SymbolKind.block))

    def gen_var_sym(self, name="anonymous") -> Symbol:
        return self.scope.put_sym(Symbol(f":{name}:", self.gen_sym_id(), SymbolKind.variable))

    def sem_expr(self, expr: Expression) -> Expression:
        return expr # TODO

    def sem_stmt(self, stmt: Stmt) -> Stmt:
        match stmt:
            case BlockDefStmt():
                if not isinstance(stmt.name, IdentExpression):
                    raise SemError(f"Only IdentExpression is allowed during block declaration")
                sym = self.gen_block_sym(stmt.name.ident)
                self.open_block()
                stmt.body = self.sem_stmt(stmt.body)
                self.close_block()
                stmt.name = SymExpression(sym)
                sym.defnode = stmt
                self._block_defs.append(stmt)
                return None
            case StmtList():
                res = []
                for inner in stmt.stmts:
                    if semmed := self.sem_stmt(inner):
                        res.append(semmed)
                return StmtList(res)
            case CallStmt():
                if isinstance(stmt.name, IdentExpression):
                    sym = self.scope.lookup_block_by_name(stmt.name.ident)
                elif isinstance(stmt.name, SymExpression):
                    sym = stmt.name.sym
                else:
                    raise SemError(f"Malformed call: {stmt}")
                stmt.name = SymExpression(sym)
                return stmt
            case CommandStmt():
                self.scope._unique_player_selectors.add(stmt.command.player_selector)
                return stmt
            case IfStmt():
                stmt.expr = self.sem_expr(stmt.expr)

                self.open_scope()
                stmt.branch_true = self.sem_stmt(stmt.branch_true)
                self.close_scope()

                self.open_scope()
                stmt.branch_false = self.sem_stmt(stmt.branch_false)
                self.close_scope()
                return stmt
            case LoopStmt():
                self.open_loop()
                stmt.body = self.sem_stmt(stmt.body)
                self.close_loop()
                return stmt
            case WhileStmt() | UntilStmt():
                stmt.expr = self.sem_expr(stmt.expr)
                self.open_loop()
                stmt.body = self.sem_stmt(stmt.body)
                self.close_loop()
                return stmt
            case TimesStmt():
                var_sym = self.gen_var_sym()
                prologue = [
                    DefVarStmt(var_sym),
                    WriteVarStmt(var_sym, NumberExpression(stmt.num)),
                ]
                epilogue = [
                    KillVarStmt(var_sym),
                ]
                cond = GreaterExpression(ReadVarExpr(SymExpression(var_sym)), NumberExpression(0))
                stmt.body.stmts.append(
                    WriteVarStmt(var_sym, SubExpression(ReadVarExpr(SymExpression(var_sym)), NumberExpression(1)))
                )

                res = StmtList(prologue + [WhileStmt(cond, stmt.body)] + epilogue)
                self.sem_stmt(res)
                return res
            case ReturnStmt():
                if self._block_nesting_level <= 0:
                    raise SemError(f"Return used outside of block scope")
                return stmt
            case BreakStmt():
                if self._loop_nesting_level <= 0:
                    raise SemError(f"Break used outside of loop scope")
                return stmt
            case DefVarStmt() | WriteVarStmt() | KillVarStmt():
                return stmt
            case _:
                raise SemError(f"Unhandled statement type: {stmt}")
        raise SemError(f"Statement fell through: {stmt}")

    def analyze_program(self):
        res = []
        for stmt in self._stmts:
            if semmed := self.sem_stmt(stmt):
                res.append(semmed)
        self._stmts = res


if __name__ == "__main__":
    from pathlib import Path

    toks = Tokenizer().tokenize(Path("./testbot.txt").read_text())
    parser = Parser(toks)
    parsed = (parser.parse())

    analyzer = Analyzer(parsed)
    analyzer.analyze_program()

    for block in analyzer._block_defs:
        print_cmd(str(block))

    for stmt in analyzer._stmts:
        print_cmd(str(stmt))
