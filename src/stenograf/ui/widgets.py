"""Shared launcher widgets: the keyboard-navigation subclasses.

Textual's defaults leave arrow keys dead or surprising in exactly the places
a form user reaches for them (PLAN.md §5 Phase 7 — the launcher is for people
who don't live in a terminal):

- a scroll container binds the arrows to *scrolling*, and those bindings sit
  between the focused field and the screen, so arrows in a form scroll the
  background instead of walking the fields;
- ``Tree`` has no left/right bindings, though right = open folder and
  left = close folder is the universal file-manager idiom;
- ``Select`` opens its menu on *up* as well as down/enter/space, which traps
  upward arrow travel at every dropdown.

Each subclass fixes one of these with a per-key binding override — the
widget's own keys are consulted before its ancestors', so anything that
genuinely uses an arrow (an open Select overlay, an Input's left/right)
still consumes it first. Screens whose whole body is scrollable prose
(Doctor, Settings) keep plain ``VerticalScroll``: there, arrows-scroll *is*
the expected behavior.
"""

from __future__ import annotations

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DirectoryTree, Select


class FormScroll(VerticalScroll):
    """Scroll container for forms: arrows move focus instead of scrolling.

    Focus-follow scrolling keeps short terminals reachable (moving focus
    scrolls the field into view), and PageUp/PageDown/Home/End still scroll.
    Non-focusable by default — focus belongs to the fields, never the box.
    """

    can_focus = False

    BINDINGS = [
        Binding("up", "app.focus_previous", "Previous field", show=False),
        Binding("down", "app.focus_next", "Next field", show=False),
    ]


class FormSelect(Select):
    """``Select`` that keeps arrow travel moving: enter/space open the menu.

    The stock widget binds enter/down/space/up all to "show the menu", so an
    arrow walk through the form gets trapped at every dropdown. Rebind the
    arrows (per-key, shadowing the stock binding) to focus movement — the
    same "arrows navigate, enter/space activate" model as every other field.
    Inside the *open* overlay the arrows belong to the option list, which is
    focused and consumes them before these bindings are consulted.
    """

    BINDINGS = [
        Binding("up", "focus_field(-1)", show=False),
        Binding("down", "focus_field(1)", show=False),
    ]

    def action_focus_field(self, delta: int) -> None:
        if delta < 0:
            self.screen.focus_previous()
        else:
            self.screen.focus_next()


class NavDirectoryTree(DirectoryTree):
    """``DirectoryTree`` with the file-manager arrow idiom.

    Right on a folder opens it (already open: steps into it); left closes
    the folder, and on a file or closed node jumps to the parent folder.
    Up/down/enter/space keep their stock Tree meanings.
    """

    BINDINGS = [
        Binding("right", "expand_or_enter", "Open folder", show=False),
        Binding("left", "collapse_or_parent", "Close folder", show=False),
    ]

    def action_expand_or_enter(self) -> None:
        node = self.cursor_node
        if node is None or not node.allow_expand:
            return  # a file: right has nowhere to go
        if node.is_expanded:
            self.action_cursor_down()
        else:
            node.expand()

    def action_collapse_or_parent(self) -> None:
        node = self.cursor_node
        if node is None:
            return
        if node.allow_expand and node.is_expanded:
            node.collapse()
        else:
            self.action_cursor_parent()
