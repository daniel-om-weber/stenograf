// A line icon drawn from SVG path data: it takes any palette colour (Qt cannot
// recolour an SVG image file without a shader pass) and stays sharp at any size.
// Paths sit on a 24×24 grid with a 2-unit stroke, after Feather Icons (MIT —
// ICONS-LICENSE next to this file). Nothing animates.
import QtQuick
import QtQuick.Shapes

Item {
    id: icon

    property string name
    property color color: Theme.text
    property real size: 16

    readonly property var paths: ({
            "mic": "M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z M19 10v2a7 7 0 0 1-14 0v-2 M12 19v4 M8 23h8",
            "mic-off": "M1 1l22 22 M9 9v3a3 3 0 0 0 5.12 2.12 M15 9.34V4a3 3 0 0 0-5.94-.6 M17 16.95A7 7 0 0 1 5 12v-2 M19 10v2a7 7 0 0 1-.11 1.23 M12 19v4 M8 23h8",
            "volume": "M11 5L6 9H2v6h4l5 4V5z M19.07 4.93a10 10 0 0 1 0 14.14 M15.54 8.46a5 5 0 0 1 0 7.07",
            "volume-off": "M11 5L6 9H2v6h4l5 4V5z M23 9l-6 6 M17 9l6 6",
            "stop": "M7 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z",
            "folder": "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
            "copy": "M11 9h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
            "check": "M20 6L9 17l-5-5"
        })
    readonly property bool filled: name === "stop"

    implicitWidth: size
    implicitHeight: size

    Shape {
        width: 24
        height: 24
        scale: icon.size / 24
        transformOrigin: Item.TopLeft
        preferredRendererType: Shape.CurveRenderer

        ShapePath {
            strokeColor: icon.color
            strokeWidth: 2
            fillColor: icon.filled ? icon.color : "transparent"
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin

            PathSvg {
                path: icon.paths[icon.name] || ""
            }
        }
    }
}
