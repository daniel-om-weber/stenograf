// A square button that is only an icon; the tooltip says what it does, so the
// row of them stays wordless. `danger` paints it red; `armed` fills it, for an
// action that asks for a second click.
import QtQuick
import QtQuick.Controls.Basic

AbstractButton {
    id: control

    property string glyph
    property string tip
    property bool danger: false
    property bool armed: false
    property bool round: false

    implicitWidth: 34
    implicitHeight: 34
    hoverEnabled: true
    opacity: enabled ? 1 : 0.4
    Accessible.role: Accessible.Button
    Accessible.name: tip

    ToolTip.visible: hovered && tip.length > 0
    ToolTip.delay: 500
    ToolTip.text: tip

    background: Rectangle {
        radius: control.round ? width / 2 : 8
        color: control.armed ? Theme.bad : control.hovered ? Theme.control : Theme.surfaceHi
        border.width: 1
        border.color: control.armed ? Theme.bad : control.activeFocus ? Theme.accent : Theme.line

        Behavior on color {
            ColorAnimation {
                duration: 90
            }
        }
    }

    contentItem: Item {
        Icon {
            anchors.centerIn: parent
            name: control.glyph
            size: Math.round(control.height * 0.48)
            color: control.armed ? Theme.bg : control.danger ? Theme.bad : Theme.text
        }
    }
}
