from typing import Any, Optional
from enum import Enum, auto

from .tokenizer import Token


class CommandKind(Enum):
    invalid = auto()

    expr = auto()
    expr_gt = auto()
    expr_eq = auto()

    kill = auto()
    sleep = auto()
    log = auto()
    teleport = auto()
    goto = auto()
    sendkey = auto()
    waitfor = auto()
    usepotion = auto()
    buypotions = auto()
    relog = auto()
    click = auto()
    tozone = auto()
    load_playstyle = auto()

class TeleportKind(Enum):
    position = auto()
    friend_icon = auto()
    friend_name = auto()
    entity_vague = auto()
    entity_literal = auto()
    mob = auto()
    quest = auto()
    client_num = auto()

class EvalKind(Enum):
    health = auto()
    max_health = auto()
    mana = auto()
    max_mana = auto()
    bagcount = auto()
    max_bagcount = auto()
    gold = auto()
    max_gold = auto()

class WaitforKind(Enum):
    dialog = auto()
    battle = auto()
    zonechange = auto()
    free = auto()
    window = auto()

class ClickKind(Enum):
    window = auto()
    position = auto()

class LogKind(Enum):
    window = auto()
    literal = auto()
    bagcount = auto()
    mana = auto()
    health = auto()
    gold = auto()

class ExprKind(Enum):
    window_visible = auto()
    in_zone = auto()
    same_zone = auto()
    playercount = auto()
    tracking_quest = auto()
    tracking_goal = auto()
    loading = auto()
    in_combat = auto()
    has_dialogue = auto()
    has_xyz = auto()
    health_below = auto()
    health_above = auto()
    health = auto()
    mana = auto()
    mana_above = auto()
    mana_below = auto()
    bag_count = auto()
    bag_count_above = auto()
    bag_count_below = auto()
    gold = auto()
    gold_above = auto()
    gold_below = auto()
    window_disabled = auto()
    same_place = auto()


# TODO: Replace asserts

class PlayerSelector:
    def __init__(self):
        self.player_nums: list[int] = []
        self.mass = False
        self.inverted = False

    def validate(self):
        assert not (self.mass and self.inverted), "Invalid player selector: mass + except"
        assert not (self.mass and len(self.player_nums) > 0), "Invalid player selector: mass + specified players"
        assert not (self.inverted and len(self.player_nums) == 0), "Invalid player selector: inverted + 0 players"
        self.player_nums.sort()

    def __hash__(self) -> int:
        return hash((frozenset(self.player_nums), self.mass, self.inverted))

    def __repr__(self) -> str:
        return f"PlayerSelector(nums: {self.player_nums}, mass: {self.mass}, inverted: {self.inverted})"

class Command:
    def __init__(self):
        self.kind = CommandKind.invalid
        self.data: list[Any] = []
        self.player_selector: PlayerSelector | None = None

    def __repr__(self) -> str:
        params_str = ", ".join([str(x) for x in self.data])
        if self.player_selector is None:
            return f"{self.kind.name}({params_str})"
        else:
            return f"{self.kind.name}({params_str}) @ {self.player_selector}"



class Expression:
    def __init__(self):
        pass

class NumberExpression(Expression):
    def __init__(self, number: float | int):
        self.number = number

    def __repr__(self) -> str:
        return f"Number({self.number})"

class StringExpression(Expression):
    def __init__(self, string: str):
        self.string = string

    def __repr__(self) -> str:
        return f"String({self.string})"

class UnaryExpression(Expression):
    def __init__(self, operator: Token, expr: Expression):
        self.operator = operator
        self.expr = expr

    def __repr__(self) -> str:
        return f"Unary({self.operator.kind}, {self.expr})"

class KeyExpression(Expression):
    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:
        return f"Key({self.key})"

class CommandExpression(Expression):
    def __init__(self, command: Command):
        self.command = command

    def __repr__(self) -> str:
        return f"ComE({self.command})"

class XYZExpression(Expression):
    def __init__(self, x: Expression, y: Expression, z: Expression):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"XYZE({self.x}, {self.y}, {self.z})"

class BinaryExpression(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

class SubExpression(BinaryExpression):
    def __repr__(self) -> str:
        return f"SubE({self.lhs}, {self.rhs})"

class DivideExpression(BinaryExpression):
    def __repr__(self) -> str:
        return f"DivideE({self.lhs}, {self.rhs})"

class EquivalentExpression(BinaryExpression):
    def __repr__(self) -> str:
        return f"EquivalentE({self.lhs}, {self.rhs})"

class GreaterExpression(BinaryExpression):
    def __repr__(self) -> str:
        return f"GreaterE({self.lhs}, {self.rhs})"

class SelectorGroup(Expression):
    def __init__(self, players: PlayerSelector, expr: Expression):
        self.players = players
        self.expr = expr

    def __repr__(self) -> str:
        return f"SelectorG({self.players}, {self.expr})"

class IdentExpression(Expression):
    def __init__(self, ident: str):
        self.ident = ident

    def __repr__(self) -> str:
        return f"IdentE({self.ident})"

class SymExpression(Expression):
    def __init__(self, sym: "Symbol"):
        self.sym = sym

    def __repr__(self) -> str:
        return f"SymE({self.sym})"

class StackLocExpression(Expression):
    def __init__(self, offset: int):
        self.offset = offset

    def __repr__(self) -> str:
        return f"StackLocE({self.offset})"

class ReadVarExpr(Expression):
    def __init__(self, loc: Expression) -> None:
        self.loc = loc

    def __repr__(self) -> str:
        return f"ReadVarE {self.loc}"


class Eval(Expression):
    def __init__(self, eval_kind: EvalKind):
        self.kind = eval_kind

    def __repr__(self) -> str:
        return f"Eval({self.kind})"

class Stmt:
    def __init__(self) -> None:
        pass

class StmtList(Stmt):
    def __init__(self, stmts: list[Stmt]):
        self.stmts = stmts

    def __repr__(self) -> str:
        return "StmtList{" + "; ".join([str(x) for x in self.stmts]) + "}"

class CommandStmt(Stmt):
    def __init__(self, command: Command):
        self.command = command

    def __repr__(self) -> str:
        return f"ComS({self.command})"

class IfStmt(Stmt):
    def __init__(self, expr: Expression, branch_true: StmtList, branch_false: StmtList):
        self.expr = expr
        self.branch_true = branch_true
        self.branch_false = branch_false

    def __repr__(self) -> str:
        return f"IfS {self.expr} {{ {self.branch_true} }} else {{ {self.branch_false} }}"

class LoopStmt(Stmt):
    def __init__(self, body: StmtList):
        self.body = body

    def __repr__(self) -> str:
        return f"LoopS {{ {self.body} }}"

class WhileStmt(Stmt):
    def __init__(self, expr: Expression, body: StmtList):
        self.expr = expr
        self.body = body

    def __repr__(self) -> str:
        return f"WhileS {self.expr} {{ {self.body} }}"

class UntilStmt(Stmt):
    def __init__(self, expr: Expression, body: StmtList):
        self.expr = expr
        self.body = body

    def __repr__(self) -> str:
        return f"UntilS {self.expr} {{ {self.body} }}"

class TimesStmt(Stmt):
    def __init__(self, num: int, body: StmtList):
        self.num = num
        self.body = body

    def __repr__(self) -> str:
        return f"TimesS {self.expr} {{ {self.body} }}"

class BlockDefStmt(Stmt):
    def __init__(self, name: Expression, body: StmtList) -> None:
        self.name = name
        self.body = body

    def __repr__(self) -> str:
        return f"BlockDefS {self.name} {{ {self.body} }}"

class CallStmt(Stmt):
    def __init__(self, name: Expression) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"CallS {self.name}"


class DefVarStmt(Stmt):
    def __init__(self, sym: "Symbol") -> None:
        self.sym = sym

    def __repr__(self) -> str:
        return f"DefVarS {self.sym}"

class WriteVarStmt(Stmt):
    def __init__(self, sym: "Symbol", expr: Expression) -> None:
        self.sym = sym
        self.expr = expr

    def __repr__(self) -> str:
        return f"WriteVarS {self.sym} = {self.expr}"

class KillVarStmt(Stmt):
    def __init__(self, sym: "Symbol") -> None:
        self.sym = sym

    def __repr__(self) -> str:
        return f"KillVarS {self.sym}"



class SymbolKind(Enum):
    variable = auto()
    block = auto()
    label = auto()


class Symbol:
    def __init__(self, literal: str, id: int, kind: SymbolKind):
        self.literal = literal
        self.id = id
        self.kind = kind
        self.defnode: Optional[Stmt] = None

    def __repr__(self) -> str:
        return f"{self.literal}:{self.id}_{self.kind.name}"
