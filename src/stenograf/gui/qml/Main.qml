// The window and the screen stack. Navigation is one-way: Python asks (the
// shell's `navigation` signal), this file acts — so no screen needs a reference
// to the StackView and every page stays a plain file with two properties.
//
// `visible` is deliberately not set here: app.py shows the window, so a headless
// test can build this whole tree — which is what catches QML errors — without a
// window ever being realized.
import QtQuick
import QtQuick.Controls.Basic

ApplicationWindow {
    id: root

    // Injected by setInitialProperties before component completion, so every
    // binding below sees it on the first evaluation (a context property does
    // NOT: it resolves late, and each Component's first pass reads null).
    property var app

    // Clamped, not fixed: 1000x680 is the intended size, but a 1280x720 logical
    // desktop (a 1080p panel at 150 % scale) is barely larger, and a smaller one
    // would have the window opening past the screen edge. A guard for those, not
    // a fix for this machine — measured on KDE 6.7.3 2026-07-25: Wayland reports
    // the *whole* panel as available (a client cannot learn panel geometry there,
    // so 720 clamps nothing), X11 reports the work area (690), and either way
    // these are content sizes while the compositor fits the decorated frame. Its
    // own shrink to 1000x690 is what actually keeps the window on screen, and
    // the layout holds at that size.
    width: Math.min(1000, Screen.desktopAvailableWidth)
    height: Math.min(680, Screen.desktopAvailableHeight)
    minimumWidth: 720
    minimumHeight: 520
    // No applicationDisplayName is set (gui/app.py says why), so this is the
    // whole window title rather than the first half of "stenograf — Stenograf".
    title: "stenograf"
    color: Theme.bg

    Connections {
        target: root.app
        function onNavigation(page, mode) {
            if (mode === "pop") {
                stack.pop();
                return;
            }
            var url = Qt.resolvedUrl(page + ".qml");
            var props = {
                app: root.app,
                screen: root.app.screen(page)
            };
            if (mode === "replace") {
                stack.replace(url, props);
            } else {
                // "root" comes from outside the window (the menu bar), which
                // knows nothing of where the stack was left: unwind to Home
                // first so the page cannot land on top of a copy of itself.
                if (mode === "root")
                    stack.pop(null);
                stack.push(url, props);
            }
        }
    }

    StackView {
        id: stack

        anchors.fill: parent
        initialItem: Home {
            app: root.app
        }

        // No page transitions: idle GPU cost for a six-item menu, and the
        // redraw budget (see gui/app.py) does not pay for decoration.
        pushEnter: null
        pushExit: null
        popEnter: null
        popExit: null
        replaceEnter: null
        replaceExit: null
    }
}
