// A button in the app's palette. Hover animates because it is event-driven and
// stops; nothing here runs while the app sits idle.
import QtQuick
import QtQuick.Controls.Basic

Button {
    id: control

    property bool primary: false

    implicitHeight: 36
    leftPadding: 16
    rightPadding: 16
    font.pixelSize: 13

    background: Rectangle {
        radius: 8
        color: !control.enabled ? Theme.surfaceHi : control.primary ? (control.hovered ? Qt.lighter(Theme.accent, 1.12) : Theme.accent) : (control.hovered ? Theme.surfaceHi : Theme.control)
        border.width: control.primary ? 0 : 1
        border.color: control.activeFocus ? Theme.accent : Theme.line

        Behavior on color {
            ColorAnimation {
                duration: 90
            }
        }
    }

    contentItem: Text {
        text: control.text
        font: control.font
        color: !control.enabled ? Theme.dim : control.primary ? Theme.accentText : Theme.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
