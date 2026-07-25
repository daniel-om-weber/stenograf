// The palette and the two type choices, in one place. A singleton so every page
// reads it without plumbing (see qmldir); nothing here is computed at runtime.
pragma Singleton
import QtQuick

QtObject {
    readonly property color bg: "#101216"
    readonly property color surface: "#191c22"
    readonly property color surfaceHi: "#20242c"
    readonly property color control: "#282d35"
    readonly property color line: "#2a2f39"
    readonly property color text: "#e7e9ee"
    readonly property color muted: "#8b93a1"
    readonly property color dim: "#5d6472"
    readonly property color accent: "#6aa2ff"
    readonly property color accentText: "#0d1117"

    // Channel colours: the live captions are channel-coarse (You / Remote)
    // until the finalize swap brings in real speaker labels. These two are the
    // app icon's two inks as well, so they move together — indigo replaced a
    // pink here on 2026-07-25 because the icon's two marks read as flames with
    // a warm second colour, and an analogous pair sits together besides.
    readonly property color mic: "#4cc9f0"
    readonly property color remote: "#8b7bf0"

    // Phase colours, keyed by the meeting screen's phase ids.
    readonly property color rec: "#ff5f56"
    readonly property color busy: "#e8b339"
    readonly property color good: "#3ddc84"
    readonly property color bad: "#ff5f56"

    readonly property string mono: Qt.platform.os === "windows" ? "Consolas"
                                 : Qt.platform.os === "osx" ? "Menlo" : "monospace"
}
