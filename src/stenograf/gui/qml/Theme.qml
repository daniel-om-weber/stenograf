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
    // 4.6:1 on `surface` and 5:1 on `bg` — the 4.5:1 that normal-size text
    // needs. It carries real reading matter (the provisional caption tail,
    // timestamps, the folder path), not only decoration.
    readonly property color dim: "#7d8594"
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
    // A muted channel: the red of a video call's struck-through mic, as a
    // dark wash with light ink so it reads as a state, not as the REC dot.
    readonly property color muteFill: "#2c1c1c"
    readonly property color muteInk: "#ff9b95"
    readonly property color bad: "#ff5f56"

    // The type scale, in logical pixels. Every text size in the app is one of
    // these, so `textScale` enlarges all of them together and the layouts —
    // which size themselves from their text — follow. Nothing smaller than 12:
    // on a 7-inch 1080p panel at 150 % an 11 px line is about 1.3 mm tall.
    readonly property real textScale: 1.0
    readonly property int textSmall: Math.round(12 * textScale)
    readonly property int textBody: Math.round(13 * textScale)
    readonly property int textLabel: Math.round(14 * textScale)
    readonly property int textReading: Math.round(15 * textScale)
    readonly property int textTitle: Math.round(17 * textScale)
    readonly property int textHeading: Math.round(20 * textScale)
    readonly property int textDisplay: Math.round(34 * textScale)

    readonly property string mono: Qt.platform.os === "windows" ? "Consolas"
                                 : Qt.platform.os === "osx" ? "Menlo" : "monospace"
}
